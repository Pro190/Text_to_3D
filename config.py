import os
from pathlib import Path
import torch
import gc
import shutil
import warnings
from typing import Dict, Any, List
import logging
from dataclasses import dataclass, field

# Принудительная очистка памяти
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Проверка доступности CUDA
if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна! Проверьте установку драйверов NVIDIA и PyTorch с поддержкой CUDA.")

# Проверка версии CUDA
cuda_version = torch.version.cuda
if cuda_version is None:
    raise RuntimeError("PyTorch не собран с поддержкой CUDA!")
print(f"Версия CUDA: {cuda_version}")

# Пути
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data" / "ModelNet10_processed"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
EXPERIMENT_DIR = BASE_DIR / "experiments"

# Создаем директории
for dir_path in [DATA_DIR, MODEL_DIR, LOGS_DIR, CHECKPOINT_DIR, EXPERIMENT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    # Параметры модели
    text_features: int = 32  # Размерность текстовых признаков
    point_features: int = 32  # Размерность признаков точек
    topology_features: int = 32  # Размерность признаков топологии
    voxel_size: int = 32  # Размер воксельной сетки
    num_points: int = 2048  # Количество точек в облаке
    max_text_length: int = 128  # Максимальная длина текстового описания
    
    # Параметры генерации
    num_generation_steps: int = 100  # Количество шагов генерации
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    
    # Гиперпараметры обучения
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Параметры early stopping и learning rate
    early_stopping_patience: int = 10  # Количество эпох без улучшения для early stopping
    learning_rate_patience: int = 5  # Количество эпох без улучшения для уменьшения learning rate
    learning_rate_factor: float = 0.5  # Множитель для уменьшения learning rate
    
    # Категории объектов
    categories: List[str] = field(default_factory=lambda: [
        'chair', 'table', 'sofa', 'bed', 'cabinet',
        'lamp', 'vase', 'plant', 'car', 'airplane'
    ])
    
    # Устройство для вычислений
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Инициализация после создания объекта"""
        # Устанавливаем количество классов на основе категорий
        self.num_classes = len(self.categories)
        
        # Проверяем, что количество классов соответствует категориям
        if self.num_classes != len(self.categories):
            raise ValueError(f"Количество классов ({self.num_classes}) не соответствует количеству категорий ({len(self.categories)})")

# Создаем объект конфигурации
config = ModelConfig()

# Экспортируем константы для обратной совместимости
TEXT_FEATURES = config.text_features
POINT_FEATURES = config.point_features
TOPOLOGY_FEATURES = config.topology_features
VOXEL_SIZE = config.voxel_size
NUM_POINTS = config.num_points
MAX_TEXT_LENGTH = config.max_text_length
NUM_GENERATION_STEPS = config.num_generation_steps
BATCH_SIZE = config.batch_size
LEARNING_RATE = config.learning_rate
NUM_EPOCHS = config.num_epochs
WEIGHT_DECAY = config.weight_decay
WARMUP_STEPS = config.warmup_steps
EARLY_STOPPING_PATIENCE = config.early_stopping_patience
LEARNING_RATE_PATIENCE = config.learning_rate_patience
LEARNING_RATE_FACTOR = config.learning_rate_factor
TEMPERATURE = config.temperature
TOP_K = config.top_k
TOP_P = config.top_p
CATEGORIES = config.categories
NUM_CLASSES = config.num_classes
DEVICE = config.device

# Пути к данным
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Параметры генерации
GENERATION_TEMPERATURE = 0.7
MAX_GENERATION_STEPS = 100
TOPOLOGY_THRESHOLD = 0.5

# Параметры валидации
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Параметры сохранения
SAVE_INTERVAL = 5
CHECKPOINT_DIR = "checkpoints"
VISUALIZATION_DIR = "visualizations"

# Параметры устройства
NUM_WORKERS = 4

# Параметры логирования
LOG_LEVEL = "INFO"
LOG_FILE = "training.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Параметры разбиения датасета
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Параметры обучения
EARLY_STOPPING_DELTA = 1e-4
LEARNING_RATE_SCHEDULER_MIN_LR = 1e-6

# Параметры сохранения
SAVE_EVERY = 5  # Сохранять модель каждые N эпох
SAVE_BEST_ONLY = True  # Сохранять только лучшую модель
CHECKPOINT_FILENAME = "model_checkpoint.pt"
BEST_MODEL_FILENAME = "best_model.pt"

# Параметры для DataLoader
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Настраиваем логирование
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Настраиваем GPU
try:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Версия CUDA: {torch.version.cuda}")
        print(f"Используется GPU: {gpu_name}")
        print(f"Доступная память GPU: {gpu_memory:.1f} GB")
    else:
        print("GPU не доступен, используется CPU")
except Exception as e:
    warnings.warn(f"Не удалось настроить GPU: {str(e)}")

# Пороговые значения для метрик
PRECISION_THRESHOLD = 0.1  # Порог для precision (расстояние в метрах)
RECALL_THRESHOLD = 0.1     # Порог для recall (расстояние в метрах)

# Конфигурации моделей
MODEL_CONFIGS = {
    "pointnet": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_delta": EARLY_STOPPING_DELTA,
        "num_points": NUM_POINTS,
        "point_features": POINT_FEATURES,
        "num_classes": NUM_CLASSES,
        "dropout": 0.1,
        "latent_dim": 128,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "accumulation_steps": 4
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Возвращает конфигурацию модели по имени"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["pointnet"]) 