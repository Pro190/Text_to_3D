import torch
import argparse
from pathlib import Path
import json
from typing import List, Optional, Union
import warnings

from model import create_model
from utils import visualize_model, load_experiment_config
from config import *

def load_model(checkpoint_path: Optional[Union[str, Path]] = None) -> torch.nn.Module:
    """Загрузка модели из чекпоинта.
    
    Args:
        checkpoint_path: Путь к чекпоинту. Если None, загружается лучшая модель.
        
    Returns:
        Загруженная модель
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_DIR / "best_model.pth"
        
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
        
    # Создаем модель
    model = create_model()
    
    # Загружаем веса
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из {checkpoint_path}")
        print(f"Эпоха: {checkpoint['epoch']}")
        print(f"Loss: {checkpoint['loss']:.4f}")
        if 'metrics' in checkpoint:
            print("Метрики:", checkpoint['metrics'])
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}")
        
    model.eval()
    return model

def generate_from_text(model: torch.nn.Module, 
                      texts: List[str],
                      output_dir: Optional[Union[str, Path]] = None) -> None:
    """Генерация 3D моделей из текстовых описаний.
    
    Args:
        model: Модель
        texts: Список текстовых описаний
        output_dir: Директория для сохранения результатов
    """
    if output_dir is None:
        output_dir = MODEL_DIR / "generated"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Генерируем
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            voxels, points = model.generate(texts)
    except Exception as e:
        raise RuntimeError(f"Ошибка при генерации: {str(e)}")
        
    # Визуализируем
    visualize_model(model, texts, epoch=-1, save_dir=output_dir)
    
    # Сохраняем метаданные
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': [
            {
                'text': t,
                'voxel_shape': v.shape,
                'points_shape': p.shape
            }
            for t, v, p in zip(texts, voxels, points)
        ],
        'model_config': load_experiment_config()
    }
    
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Результаты сохранены в {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Генерация 3D моделей из текста")
    parser.add_argument("--checkpoint", type=str, help="Путь к чекпоинту модели")
    parser.add_argument("--text", type=str, help="Текстовое описание")
    parser.add_argument("--text_file", type=str, help="Файл с текстовыми описаниями")
    parser.add_argument("--output_dir", type=str, help="Директория для сохранения результатов")
    args = parser.parse_args()
    
    # Проверяем аргументы
    if not args.text and not args.text_file:
        parser.error("Необходимо указать --text или --text_file")
        
    # Загружаем тексты
    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts.extend([line.strip() for line in f if line.strip()])
        except Exception as e:
            parser.error(f"Ошибка при чтении файла с текстами: {str(e)}")
            
    if not texts:
        parser.error("Нет текстов для генерации")
        
    # Загружаем модель
    try:
        model = load_model(args.checkpoint)
    except Exception as e:
        parser.error(f"Ошибка при загрузке модели: {str(e)}")
        
    # Генерируем
    try:
        generate_from_text(model, texts, args.output_dir)
    except Exception as e:
        parser.error(f"Ошибка при генерации: {str(e)}")

if __name__ == "__main__":
    main() 