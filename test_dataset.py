import torch
from dataset import TextTo3DDataset, verify_dataset
from config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(points: torch.Tensor, title: str = None):
    """Визуализирует облако точек"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Преобразуем в numpy
    points = points.numpy()
    
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
    
    plt.show()

def main():
    # Проверяем датасет
    print("Проверка датасета...")
    verify_dataset(DATA_DIR)
    
    # Создаем датасет
    print("\nСоздание датасета...")
    dataset = TextTo3DDataset(DATA_DIR, split='train')
    print(f"Размер датасета: {len(dataset)} сэмплов")
    
    # Получаем первый сэмпл
    print("\nЗагрузка первого сэмпла...")
    sample = dataset[0]
    print(f"Категория: {sample['category']}")
    print(f"Текст: {sample['text']}")
    print(f"Форма точек: {sample['points'].shape}")
    
    # Визуализируем точки
    print("\nВизуализация облака точек...")
    visualize_point_cloud(sample['points'], f"{sample['category']} - {sample['text']}")
    
    # Создаем DataLoader
    print("\nСоздание DataLoader...")
    dataloader = TextTo3DDataset.get_dataloader(dataset, batch_size=4)
    
    # Получаем батч
    print("\nЗагрузка батча...")
    batch = next(iter(dataloader))
    print(f"Размер батча: {batch['points'].shape}")
    print(f"Тексты: {batch['text']}")
    print(f"Категории: {batch['category']}")

if __name__ == "__main__":
    main() 