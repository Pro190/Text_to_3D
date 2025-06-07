import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import random
from config import *
import warnings
from functools import lru_cache
from transformers import BertTokenizer

def load_point_cloud(file_path: Union[str, Path]) -> np.ndarray:
    """Загружает облако точек из PLY файла.
    
    Args:
        file_path: Путь к PLY файлу
        
    Returns:
        Массив точек формы (N, 3)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
        
    try:
        # Загружаем точки из файла
        points = np.loadtxt(file_path)
        
        # Проверяем формат
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Неверный формат точек: {points.shape}")
            
        # Проверяем на NaN и Inf
        if not np.all(np.isfinite(points)):
            raise ValueError("Найдены недопустимые значения (NaN или Inf)")
            
        return points
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке {file_path}: {str(e)}")

class TextTo3DDataset(Dataset):
    """Датасет для обучения модели генерации 3D моделей из текста."""
    
    # Шаблоны описаний для каждой категории
    DESCRIPTION_TEMPLATES = {
        'bathtub': [
            "A {style} bathtub with {features}",
            "A {style} bathroom tub featuring {features}",
            "A {style} bathtub design with {features}"
        ],
        'bed': [
            "A {style} bed with {features}",
            "A {style} bedroom bed featuring {features}",
            "A {style} bed design with {features}"
        ],
        'chair': [
            "A {style} chair with {features}",
            "A {style} seating chair featuring {features}",
            "A {style} chair design with {features}"
        ],
        'desk': [
            "A {style} desk with {features}",
            "A {style} office desk featuring {features}",
            "A {style} desk design with {features}"
        ],
        'dresser': [
            "A {style} dresser with {features}",
            "A {style} bedroom dresser featuring {features}",
            "A {style} dresser design with {features}"
        ],
        'monitor': [
            "A {style} monitor with {features}",
            "A {style} computer monitor featuring {features}",
            "A {style} display design with {features}"
        ],
        'night_stand': [
            "A {style} night stand with {features}",
            "A {style} bedside table featuring {features}",
            "A {style} night stand design with {features}"
        ],
        'sofa': [
            "A {style} sofa with {features}",
            "A {style} couch featuring {features}",
            "A {style} sofa design with {features}"
        ],
        'table': [
            "A {style} table with {features}",
            "A {style} dining table featuring {features}",
            "A {style} table design with {features}"
        ],
        'toilet': [
            "A {style} toilet with {features}",
            "A {style} bathroom toilet featuring {features}",
            "A {style} toilet design with {features}"
        ]
    }
    
    # Стили для каждой категории
    STYLES = {
        'bathtub': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist'],
        'bed': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist'],
        'chair': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist', 'ergonomic'],
        'desk': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist', 'office'],
        'dresser': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist'],
        'monitor': ['modern', 'classic', 'contemporary', 'traditional', 'ultrawide', 'gaming'],
        'night_stand': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist'],
        'sofa': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist', 'sectional'],
        'table': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist', 'dining'],
        'toilet': ['modern', 'classic', 'contemporary', 'traditional', 'luxury', 'minimalist']
    }
    
    # Характеристики для каждой категории
    FEATURES = {
        'bathtub': ['curved edges', 'smooth surface', 'comfortable shape', 'ergonomic design', 'spacious interior'],
        'bed': ['comfortable mattress', 'wooden frame', 'soft headboard', 'sturdy construction', 'elegant design'],
        'chair': ['comfortable seat', 'ergonomic backrest', 'sturdy legs', 'soft padding', 'adjustable height'],
        'desk': ['spacious surface', 'sturdy legs', 'drawer storage', 'cable management', 'ergonomic design'],
        'dresser': ['multiple drawers', 'smooth surface', 'sturdy construction', 'elegant handles', 'spacious storage'],
        'monitor': ['high resolution', 'thin bezels', 'adjustable stand', 'ergonomic design', 'crisp display'],
        'night_stand': ['spacious drawer', 'smooth surface', 'sturdy legs', 'elegant design', 'convenient storage'],
        'sofa': ['comfortable cushions', 'sturdy frame', 'soft fabric', 'elegant design', 'spacious seating'],
        'table': ['smooth surface', 'sturdy legs', 'elegant design', 'spacious top', 'durable construction'],
        'toilet': ['comfortable seat', 'efficient flush', 'smooth surface', 'ergonomic design', 'modern style']
    }

    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 max_text_length: int = 77,
                 num_points: int = 2048,
                 augment: bool = True):
        """Инициализация датасета.
        
        Args:
            data_dir: Путь к директории с данными
            split: Разбиение датасета ('train', 'val' или 'test')
            max_text_length: Максимальная длина текстового описания
            num_points: Количество точек в облаке точек
            augment: Применять ли аугментацию данных
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        
        # Проверяем наличие директории
        if not data_dir.exists():
            raise FileNotFoundError(f"Директория с данными не найдена: {data_dir}")
        
        # Загружаем метаданные
        metadata_path = data_dir / "ModelNet10_processed" / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Проверяем структуру метаданных
        required_keys = ['categories', 'splits', 'statistics']
        for key in required_keys:
            if key not in self.metadata:
                raise ValueError(f"В файле метаданных отсутствует ключ: {key}")
        
        # Проверяем корректность разбиения
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Недопустимое значение split: {split}")
        
        # Сохраняем параметры
        self.data_dir = data_dir
        self.split = split
        self.max_text_length = max_text_length
        self.num_points = num_points
        self.augment = augment and split == 'train'  # Аугментация только для обучающего набора
        
        # Инициализируем токенизатор
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Загружаем список моделей для текущего разбиения
        self.models = []
        for category in self.metadata['categories']:
            category_dir = data_dir / "ModelNet10_processed" / category / split
            if not category_dir.exists():
                print(f"Предупреждение: директория {category_dir} не найдена")
                continue
                
            model_files = list(category_dir.glob("*.ply"))
            for model_file in model_files:
                self.models.append({
                    'path': model_file,
                    'category': category,
                    'text': self._get_text_description(category, model_file.stem)
                })
        
        if not self.models:
            raise RuntimeError(f"Не найдено моделей для разбиения {split}")
        
        print(f"Загружено {len(self.models)} моделей для разбиения {split}")
        print(f"Категории: {', '.join(sorted(set(m['category'] for m in self.models)))}")
        
    def __len__(self) -> int:
        return len(self.models)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Получает элемент датасета.
        
        Args:
            idx: Индекс элемента
            
        Returns:
            Словарь с точками и текстовым описанием
        """
        model = self.models[idx]
        
        # Загружаем точки
        point_path = model['path']
        points = load_point_cloud(point_path)
        
        # Проверяем количество точек
        if points.shape[0] != self.num_points:
            raise ValueError(f"Неверное количество точек в {point_path}: {points.shape[0]} != {self.num_points}")
        
        # Преобразуем в тензор
        points = torch.from_numpy(points).float()
        
        # Применяем аугментацию если включена
        if self.augment:
            # Случайное вращение
            if random.random() < 0.5:
                angle = random.uniform(0, 2 * np.pi)
                rotation_matrix = torch.tensor([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ], dtype=points.dtype)
                points = torch.matmul(points, rotation_matrix)
            
            # Случайное масштабирование
            if random.random() < 0.5:
                scale = random.uniform(0.8, 1.2)
                points = points * scale
            
            # Случайный шум
            if random.random() < 0.5:
                noise = torch.randn_like(points) * 0.02
                points = points + noise
        
        # Нормализуем точки
        points = points - points.mean(dim=0)
        points = points / (points.std(dim=0) + 1e-8)
        
        # Токенизируем текст
        text = model['text']
        text_tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'points': points,  # [NUM_POINTS, 3]
            'text': text,  # строка
            'text_tokens': text_tokens,  # словарь с токенами
            'category': model['category']  # строка
        }
        
    @staticmethod
    def get_dataloader(
        dataset: 'TextTo3DDataset',
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = PIN_MEMORY,
        persistent_workers: bool = PERSISTENT_WORKERS
    ) -> DataLoader:
        """Создает DataLoader для датасета."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

    def _get_text_description(self, category: str, model_id: str) -> str:
        """Генерирует текстовое описание для модели.
        
        Args:
            category: Категория модели
            model_id: Идентификатор модели
            
        Returns:
            Текстовое описание модели
        """
        # Используем model_id как seed для генерации
        seed = int(model_id.split('_')[-1])
        random.seed(seed)
        
        # Выбираем случайный шаблон, стиль и характеристики
        template = random.choice(self.DESCRIPTION_TEMPLATES[category])
        style = random.choice(self.STYLES[category])
        features = random.sample(self.FEATURES[category], k=random.randint(1, 3))
        features_str = ', '.join(features)
        
        # Формируем описание
        description = template.format(style=style, features=features_str)
        
        # Токенизируем и обрезаем до максимальной длины
        tokens = self.tokenizer.encode(description, 
                                     max_length=self.max_text_length,
                                     padding='max_length',
                                     truncation=True)
        
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

def create_dataset_splits(
    data_dir: Union[str, Path], 
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT
) -> None:
    """Создает разбиение датасета на train/val/test"""
    data_dir = Path(data_dir)
    
    # Проверяем метаданные
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Проверяем пропорции
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Сумма пропорций должна быть равна 1.0, получено: {total}")
    
    # Обрабатываем каждую категорию
    for category in metadata['categories']:
        category_dir = data_dir / category
        print(f"\nОбработка категории {category}...")
        
        # Получаем все файлы в категории
        all_files = []
        for split in ['train', 'val', 'test']:
            split_dir = category_dir / split
            if split_dir.exists():
                all_files.extend(list(split_dir.glob("*.ply")))
        
        if not all_files:
            print(f"Пропускаем категорию {category}: нет файлов")
            continue
            
        # Перемешиваем файлы
        random.shuffle(all_files)
        
        # Вычисляем индексы для разбиения
        n_files = len(all_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        # Разбиваем файлы
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        print(f"Всего файлов: {n_files}")
        print(f"Train: {len(train_files)}")
        print(f"Val: {len(val_files)}")
        print(f"Test: {len(test_files)}")
        
        # Создаем директории для сплитов
        for split in ["train", "val", "test"]:
            (category_dir / split).mkdir(exist_ok=True)
        
        # Перемещаем файлы
        for files, split in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
            for file in files:
                target = category_dir / split / file.name
                if not target.exists():
                    try:
                        file.rename(target)
                    except Exception as e:
                        print(f"Ошибка при перемещении {file} в {target}: {str(e)}")
                        # Если файл уже существует в целевой директории, удаляем исходный
                        if target.exists():
                            file.unlink()
    
    print("\nРазбиение датасета завершено")

def verify_dataset(data_dir: Union[str, Path]) -> None:
    """Проверяет наличие и структуру датасета.
    
    Args:
        data_dir: Путь к директории с данными
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    # Проверяем наличие директории
    if not data_dir.exists():
        raise FileNotFoundError(f"Директория с данными не найдена: {data_dir}")
    
    # Проверяем наличие файла метаданных
    metadata_path = data_dir / "ModelNet10_processed" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
    
    # Загружаем метаданные
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Проверяем структуру метаданных
    required_keys = ['categories', 'splits', 'statistics']
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"В файле метаданных отсутствует ключ: {key}")
    
    # Проверяем наличие категорий
    for category in metadata['categories']:
        category_dir = data_dir / "ModelNet10_processed" / category
        if not category_dir.exists():
            raise FileNotFoundError(f"Директория категории не найдена: {category_dir}")
        
        # Проверяем наличие поддиректорий для разбиений
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = category_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Директория разбиения {split} не найдена в {category_dir}")
            
            # Проверяем наличие файлов моделей в разбиении
            model_files = list(split_dir.glob("*.ply"))
            if not model_files:
                raise FileNotFoundError(f"В категории {category} в разбиении {split} не найдены файлы моделей")
            
            # Проверяем формат файлов
            for model_file in model_files:
                try:
                    points = load_point_cloud(model_file)
                    if points.shape != (metadata['format']['points']['num_points'], 3):
                        raise ValueError(f"Неверный формат точек в {model_file}: {points.shape}")
                    if not np.all(np.isfinite(points)):
                        raise ValueError(f"Найдены недопустимые значения в {model_file}")
                except Exception as e:
                    raise RuntimeError(f"Ошибка при проверке {model_file}: {str(e)}")
    
    print("Проверка датасета успешно завершена")
    print(f"Найдено категорий: {len(metadata['categories'])}")
    print(f"Найдено моделей: {metadata['statistics']['total_models']}")
    print("\nРаспределение по разбиениям:")
    print(f"Train: {metadata['statistics']['train_models']} моделей")
    print(f"Val: {metadata['statistics']['val_models']} моделей")
    print(f"Test: {metadata['statistics']['test_models']} моделей") 