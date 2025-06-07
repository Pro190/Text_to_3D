import torch
import numpy as np
import trimesh
import os
from config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
import wandb
import torch.nn.functional as F
import gc
from functools import lru_cache
import warnings
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

def calculate_chamfer_distance(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    batch_reduction: str = "mean"
) -> torch.Tensor:
    """Вычисление расстояния Чамфера между двумя point clouds.
    
    Args:
        pred_points: Предсказанные точки [B, N, 3]
        target_points: Целевые точки [B, M, 3]
        batch_reduction: Способ редукции по батчу ("mean" или "sum")
        
    Returns:
        Расстояние Чамфера
    """
    # Нормализуем точки
    pred_points = F.normalize(pred_points, dim=2)
    target_points = F.normalize(target_points, dim=2)
    
    # Вычисляем попарные расстояния
    dist = torch.cdist(pred_points, target_points)  # [B, N, M]
    
    # Находим минимальные расстояния в обоих направлениях
    min_dist_pred = torch.min(dist, dim=2)[0]  # [B, N]
    min_dist_target = torch.min(dist, dim=1)[0]  # [B, M]
    
    # Суммируем расстояния
    chamfer_dist = min_dist_pred.mean(dim=1) + min_dist_target.mean(dim=1)  # [B]
    
    # Редуцируем по батчу
    if batch_reduction == "mean":
        return chamfer_dist.mean()
    elif batch_reduction == "sum":
        return chamfer_dist.sum()
    else:
        raise ValueError(f"Неизвестный способ редукции: {batch_reduction}")

def calculate_metrics(
    pred_points: torch.Tensor,
    target_points: torch.Tensor
) -> Dict[str, float]:
    """Вычисление метрик для оценки качества генерации.
    
    Args:
        pred_points: Предсказанные точки [B, N, 3]
        target_points: Целевые точки [B, M, 3]
        
    Returns:
        Словарь с метриками:
        - chamfer_distance: Расстояние Чамфера между облаками точек
        - precision: Точность (доля предсказанных точек, близких к целевым)
        - recall: Полнота (доля целевых точек, близких к предсказанным)
        - f_score: F-мера (гармоническое среднее precision и recall)
    """
    # Вычисляем расстояние Чамфера
    chamfer_dist = calculate_chamfer_distance(pred_points, target_points)
    
    # Вычисляем F-счет
    precision = calculate_precision(pred_points, target_points)
    recall = calculate_recall(pred_points, target_points)
    f_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "chamfer_distance": chamfer_dist.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f_score": f_score.item()
    }

def calculate_precision(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """Вычисление precision для point cloud.
    
    Args:
        pred_points: Предсказанные точки [B, N, 3]
        target_points: Целевые точки [B, M, 3]
        
    Returns:
        Precision
    """
    # Вычисляем попарные расстояния
    dist = torch.cdist(pred_points, target_points)  # [B, N, M]
    
    # Находим минимальные расстояния для каждой предсказанной точки
    min_dist = torch.min(dist, dim=2)[0]  # [B, N]
    
    # Считаем точки, которые ближе порога
    precision = (min_dist < PRECISION_THRESHOLD).float().mean(dim=1)
    
    return precision.mean()

def calculate_recall(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """Вычисление recall для point cloud.
    
    Args:
        pred_points: Предсказанные точки [B, N, 3]
        target_points: Целевые точки [B, M, 3]
        
    Returns:
        Recall
    """
    # Вычисляем попарные расстояния
    dist = torch.cdist(target_points, pred_points)  # [B, M, N]
    
    # Находим минимальные расстояния для каждой целевой точки
    min_dist = torch.min(dist, dim=2)[0]  # [B, M]
    
    # Считаем точки, которые ближе порога
    recall = (min_dist < RECALL_THRESHOLD).float().mean(dim=1)
    
    return recall.mean()

def log_metrics(metrics: Dict[str, float], epoch: int, phase: str, writer) -> None:
    """Логирование метрик в TensorBoard и Weights & Biases.
    
    Args:
        metrics: Словарь с метриками
        epoch: Номер эпохи
        phase: Фаза ('train' или 'val')
        writer: TensorBoard writer
    """
    # Логируем в TensorBoard
    for name, value in metrics.items():
        writer.add_scalar(f"{phase}/{name}", value, epoch)
    
    # Логируем в Weights & Biases
    if wandb.run is not None:
        wandb.log({
            f"{phase}/{name}": value
            for name, value in metrics.items()
        }, step=epoch)

def clear_memory() -> None:
    """Очистка памяти GPU и CPU"""
    # Очищаем кэш CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Очищаем кэш Python
    gc.collect()

def save_experiment_config() -> None:
    """Сохранение конфигурации эксперимента"""
    config = {
        "model": {
            "text_encoder": TEXT_ENCODER_MODEL,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "num_points": NUM_POINTS,
            "point_features": POINT_FEATURES
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT
        },
        "dataset": {
            "categories": CATEGORIES,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "test_split": TEST_SPLIT
        }
    }
    
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def load_experiment_config() -> Dict:
    """Загрузка конфигурации эксперимента"""
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_model_summary(model) -> str:
    """Получение краткого описания модели"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = [
        f"Модель: {model.__class__.__name__}",
        f"Всего параметров: {total_params:,}",
        f"Обучаемых параметров: {trainable_params:,}",
        f"Текстовый энкодер: {model.text_encoder.__class__.__name__}",
        f"Декодер: {model.decoder.__class__.__name__}"
    ]
    
    return "\n".join(summary)

def visualize_point_cloud(points: torch.Tensor, title: str = "", save_path: Optional[Union[str, Path]] = None) -> None:
    """Визуализирует облако точек.
    
    Args:
        points: Тензор точек [N, 3]
        title: Заголовок графика
        save_path: Путь для сохранения графика (если None, то график отображается)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), points[:, 2].cpu().numpy(), c="b", marker=".")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if save_path is not None:
        if isinstance(save_path, str):
            save_path = Path(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def visualize_model(model, text: List[str], epoch: int, save_dir: Optional[Path] = None) -> None:
    """Визуализация выходов модели.
    
    Args:
        model: Модель
        text: Список текстовых описаний
        epoch: Номер эпохи
        save_dir: Директория для сохранения визуализаций
    """
    model.eval()
    with torch.no_grad():
        points = model.generate(text)
        points = points[0].cpu().numpy()
    
    # Ограничиваем количество сэмплов
    n_samples = min(len(text), MAX_VISUALIZE_SAMPLES)
    points = points[:n_samples]
    text = text[:n_samples]
    
    # Создаем директорию если нужно
    if save_dir is None:
        save_dir = MODEL_DIR / "visualizations"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Визуализируем каждый сэмпл
    for i, (p, t) in enumerate(zip(points, text)):
        fig = plt.figure(figsize=(10, 10))
        
        # Облако точек
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='r', marker='.')
        ax.set_title(f"Облако точек: {t}")
        
        # Сохраняем
        plt.tight_layout()
        plt.savefig(save_dir / f"epoch_{epoch}_sample_{i}.png")
        plt.close()
    
    # Сохраняем метаданные
    metadata = {
        'epoch': epoch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': [
            {
                'text': t,
                'points_shape': p.shape
            }
            for t, p in zip(text, points)
        ]
    }
    with open(save_dir / f"epoch_{epoch}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def generate_and_save(
    model: torch.nn.Module,
    text: str,
    save_dir: Path,
    prefix: str = "generated"
) -> Tuple[np.ndarray, Optional[trimesh.Trimesh]]:
    """Генерация и сохранение 3D модели из текста.
    
    Args:
        model: Модель
        text: Текстовое описание
        save_dir: Директория для сохранения
        prefix: Префикс для имен файлов
        
    Returns:
        Кортеж (points, mesh)
    """
    model.eval()
    with torch.no_grad():
        points, topology = model.generate([text])
        points = points[0].cpu().numpy()
        topology = topology[0].cpu().numpy()
    
    # Создаем директорию если нужно
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Сохраняем модель
    save_model(points, topology, save_dir, prefix)
    
    # Создаем меш для возврата
    mesh = trimesh.Trimesh(vertices=points, faces=topology)
    
    return points, mesh

def log_point_cloud(
    writer: SummaryWriter,
    points: torch.Tensor,
    tag: str,
    step: int,
    title: Optional[str] = None
) -> None:
    """Логирует облако точек в TensorBoard.
    
    Args:
        writer: TensorBoard writer
        points: Тензор с точками [N, 3]
        tag: Тег для логирования
        step: Номер шага
        title: Опциональный заголовок
    """
    # Преобразуем в numpy
    points = points.detach().cpu().numpy()
    
    # Создаем фигуру
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Отображаем точки
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
    
    # Настраиваем вид
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    
    # Устанавливаем одинаковый масштаб по осям
    max_range = max(
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    )
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    # Логируем в TensorBoard
    writer.add_figure(tag, fig, step)
    plt.close(fig)

def visualize_model(
    model: torch.nn.Module,
    texts: List[str],
    epoch: int,
    num_points: int = 2048,
    device: Optional[torch.device] = None
) -> None:
    """Визуализирует результаты модели.
    
    Args:
        model: Модель для генерации
        texts: Список текстовых описаний
        epoch: Номер эпохи
        num_points: Количество точек
        device: Устройство для вычислений
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    with torch.no_grad():
        for text in texts:
            # Генерируем точки
            points = model.generate(text, num_points, device)
            
            # Сохраняем визуализацию
            save_path = Path("visualizations") / f"epoch_{epoch}_{text}.png"
            visualize_point_cloud(points, title=text, save_path=save_path)

def calculate_metrics(
    pred_points: torch.Tensor,
    target_points: torch.Tensor
) -> Dict[str, float]:
    """Вычисляет метрики качества генерации.
    
    Args:
        pred_points: Предсказанные точки [B, N, 3]
        target_points: Целевые точки [B, N, 3]
        
    Returns:
        Словарь с метриками
    """
    # Вычисляем MSE
    mse = torch.mean((pred_points - target_points) ** 2)
    
    # Вычисляем Chamfer distance
    # TODO: Реализовать Chamfer distance
    
    return {
        'mse': mse.item(),
        'chamfer_distance': 0.0  # Временное значение
    }

def save_experiment_config() -> None:
    """Сохраняет конфигурацию эксперимента"""
    config = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'latent_dim': LATENT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'num_points': NUM_POINTS,
        'weight_decay': WEIGHT_DECAY,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'early_stopping_delta': EARLY_STOPPING_DELTA,
        'learning_rate_scheduler_factor': LEARNING_RATE_SCHEDULER_FACTOR,
        'learning_rate_scheduler_patience': LEARNING_RATE_SCHEDULER_PATIENCE,
        'learning_rate_scheduler_min_lr': LEARNING_RATE_SCHEDULER_MIN_LR
    }
    
    # Сохраняем в JSON
    with open(EXPERIMENT_DIR / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def get_model_summary(model: torch.nn.Module) -> str:
    """Возвращает текстовое описание архитектуры модели"""
    summary = []
    total_params = 0
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        summary.append(f"{name}: {params:,} параметров")
    
    summary.append(f"\nВсего параметров: {total_params:,}")
    return "\n".join(summary)

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str,
    writer: Optional[SummaryWriter] = None
) -> None:
    """Логирует метрики в TensorBoard и wandb"""
    # Логируем в TensorBoard
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(f"{prefix}/{name}", value, step)
    
    # Логируем в wandb
    wandb.log({f"{prefix}/{name}": value for name, value in metrics.items()}, step=step)

def clear_memory() -> None:
    """Очищает память GPU и CPU"""
    import gc
    
    # Очищаем кэш CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Запускаем сборщик мусора
    gc.collect()

def save_as_glb(mesh: trimesh.Trimesh, save_path: Union[str, Path]) -> None:
    """Сохраняет меш в формате GLB.
    
    Args:
        mesh: Меш для сохранения
        save_path: Путь для сохранения
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    mesh.export(save_path, file_type='glb')

def voxels_to_mesh(voxels: torch.Tensor) -> trimesh.Trimesh:
    """Преобразует воксели в меш.
    
    Args:
        voxels: Тензор вокселей [N, N, N]
        
    Returns:
        Меш из вокселей
    """
    # Преобразуем в numpy
    voxels = voxels.detach().cpu().numpy()
    
    # Создаем меш из вокселей
    vertices, faces, _, _ = trimesh.voxel.matrix_to_marching_cubes(voxels)
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def points_to_mesh(points: torch.Tensor) -> trimesh.Trimesh:
    """Преобразует точки в меш.
    
    Args:
        points: Тензор точек [N, 3]
        
    Returns:
        Меш из точек
    """
    # Преобразуем в numpy
    points = points.detach().cpu().numpy()
    
    # Создаем меш из точек
    return trimesh.Trimesh(vertices=points)

def validate_mesh(mesh: trimesh.Trimesh) -> Tuple[bool, List[str]]:
    """Проверяет валидность меша.
    
    Args:
        mesh: Меш для проверки
        
    Returns:
        Кортеж (is_valid, errors):
        - is_valid: Флаг валидности меша
        - errors: Список найденных ошибок
    """
    errors = []
    
    # Проверяем наличие дегенеративных треугольников
    if not mesh.is_watertight:
        errors.append("Меш не является водонепроницаемым")
    
    # Проверяем согласованность нормалей
    if not mesh.is_winding_consistent:
        errors.append("Нормали не согласованы")
        
    # Проверяем наличие дегенеративных треугольников
    degenerate_faces = mesh.nondegenerate_faces()
    if not np.all(degenerate_faces):
        errors.append("Обнаружены дегенеративные треугольники")
        
    # Проверяем наличие самопересечений через проверку компонентов
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        errors.append("Обнаружены самопересечения (меш разбит на несколько компонентов)")
        
    return len(errors) == 0, errors

def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Ремонтирует меш, исправляя основные проблемы.
    
    Args:
        mesh: Меш для ремонта
        
    Returns:
        Отремонтированный меш
    """
    # Удаляем дегенеративные треугольники
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    
    # Исправляем нормали
    if not mesh.is_winding_consistent:
        mesh.fix_normals()
    
    # Заполняем дыры
    if not mesh.is_watertight:
        # Сначала пытаемся заполнить дыры
        mesh.fill_holes()
        
        # Если все еще есть дыры, используем более агрессивный метод
        if not mesh.is_watertight:
            # Создаем выпуклую оболочку
            hull = mesh.convex_hull
            
            # Объединяем с исходным мешем
            mesh = trimesh.util.concatenate([mesh, hull])
            
            # Удаляем дубликаты вершин
            mesh.process(validate=True)
            
            # Снова заполняем дыры
            mesh.fill_holes()
    
    # Исправляем самопересечения
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # Берем самый большой компонент
        mesh = max(components, key=lambda x: len(x.faces))
        
        # Обрабатываем его
        mesh.process(validate=True)
        
        # Пытаемся сделать его водонепроницаемым
        if not mesh.is_watertight:
            hull = mesh.convex_hull
            mesh = trimesh.util.concatenate([mesh, hull])
            mesh.process(validate=True)
            mesh.fill_holes()
    
    return mesh

def save_model(
    points: Union[torch.Tensor, np.ndarray],
    topology: Union[torch.Tensor, np.ndarray], 
    save_path: Union[str, Path],
    prefix: str = "generated"
) -> None:
    """Сохраняет 3D модель в формате PLY.
    
    Args:
        points: Тензор или массив точек [N, 3]
        topology: Тензор или массив индексов вершин для граней [F, 3]
        save_path: Путь для сохранения (директория)
        prefix: Префикс для имен файлов
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
        
    # Создаем директорию если нужно
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Конвертируем тензоры в numpy массивы если нужно
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if isinstance(topology, torch.Tensor):
            topology = topology.detach().cpu().numpy()
            
        # Создаем меш из точек и топологии
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=topology,
            process=True  # Включаем обработку для исправления проблем
        )
        
        # Валидируем меш
        is_valid, errors = validate_mesh(mesh)
        if not is_valid:
            warnings.warn(f"Меш не прошел валидацию: {', '.join(errors)}")
            
            # Пытаемся отремонтировать меш
            mesh = repair_mesh(mesh)
            
            # Повторная валидация после ремонта
            is_valid, errors = validate_mesh(mesh)
            if not is_valid:
                warnings.warn(f"Не удалось полностью исправить проблемы меша: {', '.join(errors)}")
                warnings.warn("Сохраняем меш в текущем состоянии")
        
        # Формируем пути для сохранения
        ply_path = save_path / f"{prefix}.ply"
        vis_path = save_path / f"{prefix}_vis.png"
        
        # Сохраняем в PLY
        mesh.export(str(ply_path))
        
        # Создаем визуализацию
        visualize_mesh(mesh, save_path=str(vis_path), show=False)
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при сохранении меша: {str(e)}")

def visualize_mesh(
    mesh_or_points: Union[trimesh.Trimesh, torch.Tensor, np.ndarray],
    faces: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """Визуализирует 3D меш или облако точек.
    
    Args:
        mesh_or_points: Меш trimesh или тензор/массив точек [N, 3]
        faces: Тензор/массив индексов вершин для граней [F, 3] (опционально)
        save_path: Путь для сохранения визуализации (опционально)
        show: Показывать ли визуализацию (по умолчанию True)
    """
    try:
        import pyrender
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Для визуализации требуется установить pyrender и matplotlib")
        
    # Конвертируем входные данные в нужный формат
    if isinstance(mesh_or_points, trimesh.Trimesh):
        mesh = mesh_or_points
    else:
        # Конвертируем тензоры в numpy массивы
        if isinstance(mesh_or_points, torch.Tensor):
            points = mesh_or_points.detach().cpu().numpy()
        else:
            points = np.asarray(mesh_or_points)
            
        if faces is not None:
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()
            else:
                faces = np.asarray(faces)
                
            # Создаем меш из точек и граней
            mesh = trimesh.Trimesh(vertices=points, faces=faces)
        else:
            # Если грани не заданы, создаем меш из выпуклой оболочки точек
            mesh = trimesh.convex.convex_hull(points)
    
    # Создаем сцену
    scene = pyrender.Scene()
    
    # Добавляем меш в сцену
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0],
        metallicFactor=0.2,
        roughnessFactor=0.8
    )
    
    mesh_render = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh_render)
    
    # Настраиваем камеру
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Добавляем освещение
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(light, pose=light_pose)
    
    # Рендерим сцену
    r = pyrender.OffscreenRenderer(640, 480)
    color, _ = r.render(scene)
    
    # Сохраняем или показываем результат
    if save_path is not None:
        if isinstance(save_path, str):
            save_path = Path(save_path)
        plt.imsave(str(save_path), color)
        
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(color)
        plt.axis('off')
        plt.show()
        
    plt.close()

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
            text = batch['text']
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            pred_points = model(text)
            
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
                text = batch['text']
                
                pred_points = model(text)
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
            text = batch['text']
            
            pred_points = model(text)
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