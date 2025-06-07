import torch
import argparse
import os
from pathlib import Path
from typing import Union, List, Dict, Optional
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import sys
import time
import logging
import numpy as np

from model import create_model
from utils import (
    save_model,
    generate_and_save,
    visualize_mesh
)
from config import (
    DEVICE,
    MODEL_DIR,
    NUM_POINTS,
    POINT_FEATURES,
    CATEGORIES,
    ModelConfig
)

# Добавляем ModelConfig в список безопасных глобальных объектов
torch.serialization.add_safe_globals([ModelConfig])

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_available_categories():
    """Возвращает список доступных категорий"""
    return CATEGORIES

def print_categories():
    """Выводит список доступных категорий"""
    print("\nДоступные категории предметов:")
    for i, category in enumerate(CATEGORIES, 1):
        print(f"{i}. {category}")

def get_user_input() -> str:
    """Получает и валидирует ввод пользователя"""
    while True:
        print("\nВыберите действие:")
        print("1. Выбрать из списка категорий")
        print("2. Ввести свой текст")
        print("3. Выйти")
        
        choice = input("\nВаш выбор (1-3): ").strip()
        
        if choice == "1":
            print_categories()
            while True:
                try:
                    cat_num = int(input("\nВведите номер категории: ").strip())
                    if 1 <= cat_num <= len(CATEGORIES):
                        return get_available_categories()[cat_num - 1]
                    print(f"Пожалуйста, введите число от 1 до {len(CATEGORIES)}")
                except ValueError:
                    print("Пожалуйста, введите корректный номер")
        
        elif choice == "2":
            text = input("\nВведите описание предмета: ").strip()
            if text:
                return text
            print("Описание не может быть пустым")
        
        elif choice == "3":
            print("\nДо свидания!")
            sys.exit(0)
        
        else:
            print("Пожалуйста, выберите 1, 2 или 3")

def find_latest_model(model_dir: Union[str, Path] = MODEL_DIR) -> Optional[Path]:
    """Находит последнюю сохраненную модель в директории"""
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)
    
    model_files = list(model_dir.glob("*.pth"))
    if not model_files:
        return None
    
    # Сортируем по времени модификации
    return max(model_files, key=lambda x: x.stat().st_mtime)

def load_model(model_path: str) -> torch.nn.Module:
    """Загружает модель из файла"""
    try:
        # Создаем экземпляр модели с текущей конфигурацией
        model = create_model().to(DEVICE)
        
        # Загружаем состояние модели с отключенным weights_only
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Проверяем формат сохраненного состояния
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Преобразуем ключи состояния для соответствия новой архитектуре
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('text_encoder.bert.'):
                new_key = k.replace('text_encoder.bert.', 'text_encoder.')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
                
        # Загружаем состояние с игнорированием отсутствующих ключей
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}")

def generate_model(text: str, model_path: str, output_dir: str = "results") -> Dict[str, str]:
    """Генерирует 3D модель из текстового описания"""
    try:
        # Проверяем входные параметры
        if not text or not isinstance(text, str):
            raise ValueError("Текст должен быть непустой строкой")
        if not os.path.exists(model_path):
            raise ValueError(f"Файл модели не найден: {model_path}")
            
        # Создаем уникальную директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(output_dir, f"generation_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Загружаем модель
        model = load_model(model_path)
        
        # Генерируем модель
        with torch.no_grad():
            points, topology = model.generate([text])
            
        # Сохраняем результаты
        model_path = os.path.join(result_dir, "model.ply")
        save_model(points[0], topology[0], model_path)
        
        # Создаем визуализацию
        vis_path = os.path.join(result_dir, "visualization.png")
        visualize_mesh(points[0], topology[0], vis_path)
        
        # Сохраняем метаданные
        metadata = {
            "text": text,
            "timestamp": timestamp,
            "model_path": model_path,
            "visualization_path": vis_path
        }
        metadata_path = os.path.join(result_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        return {
            "model_path": model_path,
            "visualization_path": vis_path,
            "metadata_path": metadata_path,
            "metadata": metadata
        }
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при генерации модели: {str(e)}")

def interactive_mode(model_path: Union[str, Path] = "checkpoints/best_model.pth"):
    """Интерактивный режим генерации моделей"""
    print("\n=== Генератор 3D моделей ===")
    print("Добро пожаловать в интерактивный режим генерации!")
    
    try:
        # Пробуем загрузить модель при старте
        model = load_model(model_path)
    except Exception as e:
        print(f"\nОшибка при загрузке модели: {str(e)}")
        print("Проверьте наличие файла модели и его совместимость.")
        return
    
    while True:
        try:
            text = get_user_input()
            generate_model(text, model_path)
            
            print("\nХотите сгенерировать еще одну модель? (да/нет)")
            if input().lower().strip() not in ['да', 'y', 'yes']:
                print("\nСпасибо за использование генератора!")
                break
                
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем")
            break
        except Exception as e:
            print(f"\nПроизошла ошибка: {str(e)}")
            print("Попробуйте еще раз или выберите другой вариант")

def main():
    parser = argparse.ArgumentParser(description="Генерация 3D моделей из текста")
    parser.add_argument("--text", type=str, help="Текстовое описание модели")
    parser.add_argument("--text_file", type=str, help="Файл со списком текстовых описаний")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Путь к обученной модели")
    parser.add_argument("--output", type=str, default="results", help="Директория для сохранения результатов")
    parser.add_argument("--interactive", action="store_true", help="Запустить в интерактивном режиме")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.model)
        return
    
    if not args.text and not args.text_file:
        print("Запуск в интерактивном режиме...")
        interactive_mode(args.model)
        return
    
    if args.text_file:
        try:
            with open(args.text_file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
            for text in texts:
                generate_model(text, args.model, args.output)
        except Exception as e:
            warnings.warn(f"Ошибка при чтении файла с текстами: {str(e)}")
            raise
    else:
        generate_model(args.text, args.model, args.output)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка: {str(e)}")
        sys.exit(1) 