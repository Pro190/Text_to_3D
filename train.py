import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import os
import gc
import signal
import sys
from datetime import datetime, timedelta
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from model import TextTo3DModel
from dataset import TextTo3DDataset, verify_dataset
from utils import (
    calculate_metrics, visualize_model, save_experiment_config,
    get_model_summary, log_metrics, clear_memory, log_point_cloud,
    visualize_point_cloud, train_model, test_model
)
from config import *
from losses import ChamferDistance

os.environ["WANDB_MODE"] = "disabled"

# Устанавливаем случайные зерна для воспроизводимости
def set_seed(seed: int = 42):
    """Устанавливает случайные зерна для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Устанавливаем зерно
set_seed()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        self.verbose = verbose
        
    def __call__(self, value):
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.should_stop = True
            
        if self.should_stop and self.verbose:
            print(f"Early stopping сработал! {self.counter}/{self.patience}")
            
        return self.should_stop

class TrainingInterrupted(Exception):
    """Исключение для обработки прерывания обучения"""
    pass

def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    raise TrainingInterrupted("Обучение прервано пользователем")

# Регистрируем обработчик сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    is_best: bool = False
):
    """Сохраняет чекпоинт модели"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    # Сохраняем последний чекпоинт
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Если это лучшая модель, сохраняем отдельно
    if is_best:
        best_path = CHECKPOINT_DIR / "best_model.pt"
        torch.save(checkpoint, best_path)
        
        # Удаляем старые чекпоинты
        for old_checkpoint in CHECKPOINT_DIR.glob("checkpoint_epoch_*.pt"):
            if old_checkpoint != checkpoint_path:
                try:
                    old_checkpoint.unlink()
                except Exception as e:
                    warnings.warn(f"Не удалось удалить старый чекпоинт {old_checkpoint}: {str(e)}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Загрузка чекпоинта модели"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    except Exception as e:
        warnings.warn(f"Не удалось загрузить чекпоинт: {str(e)}")
        return 0, {}

def clear_memory():
    """Расширенная очистка памяти"""
    # Очищаем кэш CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Принудительно запускаем сборщик мусора
    gc.collect()

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Обучает модель на одной эпохе"""
    model.train()
    total_loss = 0
    total_point_loss = 0
    total_cosine_loss = 0
    total_samples = 0
    all_metrics = []
    
    # Создаем прогресс-бар
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Переносим данные на GPU
        points = batch['points'].to(device)  # [B, N, 3]
        texts = batch['text']
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        try:
            # Прямой проход
            output = model(points, texts)
            
            # Проверяем выход модели
            if torch.isnan(output).any():
                raise ValueError("Модель выдала NaN значения")
                
            # Вычисляем функцию потерь
            # Используем MSE для точек и косинусное расстояние для текста
            point_loss = F.mse_loss(output, points)
            
            # Нормализуем выходы для косинусного расстояния
            output_norm = F.normalize(output, p=2, dim=-1)
            points_norm = F.normalize(points, p=2, dim=-1)
            cosine_loss = 1 - F.cosine_similarity(output_norm, points_norm).mean()
            
            # Комбинируем потери
            loss = point_loss + 0.1 * cosine_loss
            
            # Обратный проход
            loss.backward()
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            
            # Обновляем веса
            optimizer.step()
            
            # Вычисляем метрики
            metrics = calculate_metrics(output, points)
            all_metrics.append(metrics)
            
            # Обновляем статистику
            batch_size = points.size(0)
            total_loss += loss.item() * batch_size
            total_point_loss += point_loss.item() * batch_size
            total_cosine_loss += cosine_loss.item() * batch_size
            total_samples += batch_size
            
            # Обновляем прогресс-бар
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'point_loss': f"{point_loss.item():.4f}",
                'cosine_loss': f"{cosine_loss.item():.4f}",
                'chamfer': f"{metrics['chamfer_distance']:.4f}",
                'avg_loss': f"{total_loss/total_samples:.4f}"
            })
            
            # Логируем в TensorBoard
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + pbar.n)
                writer.add_scalar('train/point_loss', point_loss.item(), epoch * len(dataloader) + pbar.n)
                writer.add_scalar('train/cosine_loss', cosine_loss.item(), epoch * len(dataloader) + pbar.n)
                for name, value in metrics.items():
                    writer.add_scalar(f'train/{name}', value, epoch * len(dataloader) + pbar.n)
                
        except Exception as e:
            print(f"Ошибка при обработке батча: {str(e)}")
            continue
    
    # Вычисляем средние метрики
    metrics = {
        'loss': total_loss / total_samples if total_samples > 0 else float('inf'),
        'point_loss': total_point_loss / total_samples if total_samples > 0 else float('inf'),
        'cosine_loss': total_cosine_loss / total_samples if total_samples > 0 else float('inf')
    }
    
    # Добавляем средние метрики по батчам
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        metrics.update(avg_metrics)
    
    return metrics

def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Валидирует модель"""
    model.eval()
    total_loss = 0
    total_point_loss = 0
    total_cosine_loss = 0
    total_samples = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            try:
                # Переносим данные на GPU
                points = batch['points'].to(device)
                texts = batch['text']
                
                # Прямой проход
                output = model(points, texts)
                
                # Проверяем выход модели
                if torch.isnan(output).any():
                    raise ValueError("Модель выдала NaN значения")
                
                # Вычисляем функцию потерь
                point_loss = F.mse_loss(output, points)
                
                # Нормализуем выходы для косинусного расстояния
                output_norm = F.normalize(output, p=2, dim=-1)
                points_norm = F.normalize(points, p=2, dim=-1)
                cosine_loss = 1 - F.cosine_similarity(output_norm, points_norm).mean()
                
                # Комбинируем потери
                loss = point_loss + 0.1 * cosine_loss
                
                # Вычисляем метрики
                metrics = calculate_metrics(output, points)
                all_metrics.append(metrics)
                
                # Обновляем статистику
                batch_size = points.size(0)
                total_loss += loss.item() * batch_size
                total_point_loss += point_loss.item() * batch_size
                total_cosine_loss += cosine_loss.item() * batch_size
                total_samples += batch_size
                
            except Exception as e:
                print(f"Ошибка при валидации батча: {str(e)}")
                continue
    
    # Вычисляем средние метрики
    metrics = {
        'loss': total_loss / total_samples if total_samples > 0 else float('inf'),
        'point_loss': total_point_loss / total_samples if total_samples > 0 else float('inf'),
        'cosine_loss': total_cosine_loss / total_samples if total_samples > 0 else float('inf')
    }
    
    # Добавляем средние метрики по батчам
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        metrics.update(avg_metrics)
    
    # Логируем в TensorBoard
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(f'val/{name}', value, epoch)
    
    return metrics

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    writer: Optional[SummaryWriter] = None
) -> None:
    """Обучает модель"""
    # Создаем оптимизатор и планировщик
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Создаем функцию потерь
    criterion = nn.MSELoss()
    
    # Инициализируем scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LEARNING_RATE_SCHEDULER_FACTOR,
        patience=LEARNING_RATE_SCHEDULER_PATIENCE,
        min_lr=LEARNING_RATE_SCHEDULER_MIN_LR
    )
    
    # Инициализируем early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_DELTA,
        verbose=True
    )
    
    # Инициализируем метрики
    best_val_loss = float('inf')
    val_metrics = {'loss': float('inf')}
    
    # Настраиваем обработчик прерывания
    def signal_handler(signum, frame):
        raise TrainingInterrupted()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Обучаем модель
        for epoch in range(NUM_EPOCHS):
            print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")
            
            # Обучаем на одной эпохе
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, writer)
            
            # Валидируем
            val_metrics = validate(model, val_loader, criterion, DEVICE, epoch, writer)
            
            # Обновляем learning rate
            scheduler.step(val_metrics['loss'])
            
            # Проверяем early stopping
            if early_stopping(val_metrics['loss']):
                print("Early stopping сработал!")
                break
            
            # Сохраняем лучшую модель
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(model, optimizer, epoch, val_metrics, True)
            else:
                save_checkpoint(model, optimizer, epoch, val_metrics, False)
            
            # Очищаем память
            clear_memory()
            
    except TrainingInterrupted:
        print("\nОбучение прервано пользователем")
        # Сохраняем последнюю модель при прерывании
        save_checkpoint(model, optimizer, epoch, val_metrics, False)
    except Exception as e:
        print(f"\nОшибка при обучении: {str(e)}")
        # Сохраняем последнюю модель при ошибке
        save_checkpoint(model, optimizer, epoch, val_metrics, False)
        raise
    finally:
        # Закрываем writer
        if writer is not None:
            writer.close()

def main():
    """Основная функция для обучения модели."""
    # Проверяем наличие датасета
    verify_dataset(DATA_DIR)
    
    print("Создание датасетов...")
    train_dataset = TextTo3DDataset(DATA_DIR, split='train')
    val_dataset = TextTo3DDataset(DATA_DIR, split='val')
    test_dataset = TextTo3DDataset(DATA_DIR, split='test')
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print("Создание модели...")
    # Создаем конфигурацию модели
    config = ModelConfig(
        device=DEVICE,
        num_points=train_dataset.num_points,
        max_text_length=train_dataset.max_text_length,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Создаем модель с конфигурацией
    model = TextTo3DModel(config).to(DEVICE)
    
    # Создаем оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Создаем планировщик скорости обучения
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Создаем функцию потерь
    criterion = ChamferDistance()
    
    # Создаем writer для TensorBoard
    writer = SummaryWriter(log_dir=LOGS_DIR)
    
    print("Начало обучения...")
    # Обучаем модель
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config,
        writer=writer
    )
    
    print("Тестирование модели...")
    # Тестируем модель
    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=config.device,
        writer=writer
    )
    
    # Закрываем writer
    writer.close()
    
    print("\nОбучение завершено!")
    print(f"Лучшая валидационная потеря: {min(history['val_loss']):.4f}")
    print(f"Тестовая потеря: {test_metrics['test_loss']:.4f}")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    config: ModelConfig,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, List[float]]:
    """Обучает модель.
    
    Args:
        model: Модель для обучения
        train_loader: Загрузчик обучающих данных
        val_loader: Загрузчик валидационных данных
        optimizer: Оптимизатор
        scheduler: Планировщик скорости обучения
        criterion: Функция потерь
        config: Конфигурация модели
        writer: Writer для TensorBoard (опционально)
        
    Returns:
        Словарь с историей метрик
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Создаем директорию для чекпоинтов
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(config.num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            # Перемещаем данные на устройство
            points = batch['points'].to(config.device)
            text_tokens = {k: v.to(config.device) for k, v in batch['text_tokens'].items()}
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            pred_points = model(text_tokens)
            
            # Вычисляем потери
            loss = criterion(pred_points, points)
            
            # Обратный проход
            loss.backward()
            
            # Обновляем веса
            optimizer.step()
            
            # Собираем статистику
            train_loss += loss.item()
        
        # Вычисляем средние потери
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(config.device)
                text_tokens = {k: v.to(config.device) for k, v in batch['text_tokens'].items()}
                
                pred_points = model(text_tokens)
                loss = criterion(pred_points, points)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Обновляем скорость обучения
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # Логируем метрики
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)
        
        print(f'Epoch {epoch + 1}/{config.num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Сохраняем чекпоинт
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f'Сохранена лучшая модель с val_loss = {val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f'Ранняя остановка на эпохе {epoch + 1}')
                break
    
    return history

def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Тестирует модель.
    
    Args:
        model: Обученная модель
        test_loader: Загрузчик тестовых данных
        criterion: Функция потерь
        device: Устройство для вычислений
        writer: Writer для TensorBoard (опционально)
        
    Returns:
        Словарь с метриками на тестовом наборе
    """
    model.eval()
    test_loss = 0.0
    metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            points = batch['points'].to(device)
            text_tokens = {k: v.to(device) for k, v in batch['text_tokens'].items()}
            
            pred_points = model(text_tokens)
            loss = criterion(pred_points, points)
            test_loss += loss.item()
            
            # Вычисляем дополнительные метрики
            chamfer_dist = criterion(pred_points, points)
            metrics['chamfer_distance'] = chamfer_dist.item()
    
    # Вычисляем средние метрики
    test_loss /= len(test_loader)
    metrics['test_loss'] = test_loss
    
    # Логируем результаты
    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, 0)
        for name, value in metrics.items():
            writer.add_scalar(f'Metrics/{name}', value, 0)
    
    print("\nРезультаты на тестовом наборе:")
    print(f'Test Loss: {test_loss:.4f}')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')
    
    return metrics

if __name__ == "__main__":
    main() 