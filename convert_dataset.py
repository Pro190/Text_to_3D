import os
import trimesh
from pathlib import Path
from tqdm import tqdm
import numpy as np
from config import *

def convert_off_to_ply(input_path: Path, output_path: Path, num_points: int = NUM_POINTS):
    """Конвертирует .off файл в .ply с фиксированным количеством точек"""
    try:
        # Загружаем меш
        mesh = trimesh.load(input_path)
        
        # Сэмплируем точки с поверхности меша
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        
        # Создаем директорию если нужно
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем точки в .ply
        np.savetxt(output_path, points, fmt="%.6f")
        
        return True
    except Exception as e:
        print(f"Ошибка при конвертации {input_path}: {str(e)}")
        return False

def process_category(category_dir: Path, output_dir: Path):
    """Обрабатывает все файлы в категории"""
    # Создаем директории для сплитов
    for split in ["train", "val", "test"]:
        (output_dir / category_dir.name / split).mkdir(parents=True, exist_ok=True)
    
    # Получаем список всех .off файлов
    off_files = []
    for split in ["train", "test"]:
        split_dir = category_dir / split
        if split_dir.exists():
            off_files.extend(list(split_dir.glob("*.off")))
    
    # Конвертируем файлы
    success_count = 0
    for off_file in tqdm(off_files, desc=f"Обработка {category_dir.name}"):
        # Определяем сплит
        split = off_file.parent.name
        
        # Создаем путь для .ply файла
        ply_file = output_dir / category_dir.name / split / f"{off_file.stem}.ply"
        
        # Конвертируем
        if convert_off_to_ply(off_file, ply_file):
            success_count += 1
    
    return success_count

def main():
    # Пути к директориям
    input_dir = Path("data/ModelNet10")
    output_dir = Path("data/ModelNet10_processed")
    
    # Создаем корневую директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем список категорий
    categories = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    # Обрабатываем каждую категорию
    total_files = 0
    for category in categories:
        if category.name in ["__MACOSX", "__pycache__"]:
            continue
            
        print(f"\nОбработка категории: {category.name}")
        success_count = process_category(category, output_dir)
        total_files += success_count
        print(f"Успешно конвертировано: {success_count} файлов")
    
    print(f"\nВсего конвертировано файлов: {total_files}")

if __name__ == "__main__":
    main() 