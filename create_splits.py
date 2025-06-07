import os
from pathlib import Path
import shutil
import random
from dataset import create_dataset_splits, verify_dataset
from config import *

def main():
    # Проверяем датасет
    print("Проверка датасета...")
    verify_dataset(DATA_DIR)
    
    # Создаем разбиение
    print("\nСоздание разбиения датасета...")
    create_dataset_splits(
        DATA_DIR,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        test_ratio=TEST_SPLIT
    )
    
    # Проверяем результат
    print("\nПроверка разбиения...")
    verify_dataset(DATA_DIR)
    
    print("\nРазбиение датасета завершено")

if __name__ == "__main__":
    main() 