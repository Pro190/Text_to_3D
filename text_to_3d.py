import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import os

# === 1. Загрузка датасета ===
def load_dataset(dataset_dir):
    dataset = []
    max_vertices = 0
    max_faces = 0
    
    # Первый проход: находим максимальное количество вершин и граней
    for obj_file in Path(dataset_dir).glob('*.obj'):
        mesh = trimesh.load(obj_file, force='mesh')
        max_vertices = max(max_vertices, len(mesh.vertices))
        max_faces = max(max_faces, len(mesh.faces))
    
    print(f"Максимальное количество вершин: {max_vertices}")
    print(f"Максимальное количество граней: {max_faces}")
    
    # Второй проход: загружаем данные
    for obj_file in Path(dataset_dir).glob('*.obj'):
        name = obj_file.stem.lower()
        mesh = trimesh.load(obj_file, force='mesh')
        
        # Нормализуем вершины
        vertices = mesh.vertices.astype(np.float32)
        center = vertices.mean(axis=0)
        vertices = vertices - center
        scale = np.abs(vertices).max()
        if scale > 0:
            vertices = vertices / scale
            
        # Паддинг вершин до максимального размера
        padded_vertices = np.zeros((max_vertices, 3), dtype=np.float32)
        padded_vertices[:len(vertices)] = vertices
        
        # Паддинг граней до максимального размера
        padded_faces = np.zeros((max_faces, 3), dtype=np.int32)
        padded_faces[:len(mesh.faces)] = mesh.faces
        
        dataset.append({
            'name': name,
            'vertices': padded_vertices,
            'faces': padded_faces,
            'num_vertices': len(vertices),
            'num_faces': len(mesh.faces)
        })
        print(f"Загружен {name}: {len(vertices)} вершин, {len(mesh.faces)} граней")
    
    return dataset, max_vertices, max_faces

# === 2. Модель ===
class TextTo3DModel(nn.Module):
    def __init__(self, max_vertices, max_faces):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = self.bert.to(self.device)  # Перемещаем BERT на нужное устройство
        
        # Размерность выходных признаков BERT
        bert_output_dim = 768
        
        # Декодер для вершин
        self.vertex_decoder = nn.Sequential(
            nn.Linear(bert_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, max_vertices * 3)  # x, y, z для каждой вершины
        )
        
        # Декодер для граней
        self.face_decoder = nn.Sequential(
            nn.Linear(bert_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, max_faces * 3)  # 3 индекса для каждой грани
        )
    
    def encode_text(self, text):
        # Токенизация текста
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Перемещаем входные данные на нужное устройство
        
        # Получаем эмбеддинги текста
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Берем [CLS] токен
    
    def forward(self, text):
        text_features = self.encode_text(text)
        vertices = self.vertex_decoder(text_features)
        faces = self.face_decoder(text_features)
        
        # Преобразуем выходы в нужную форму
        vertices = vertices.view(-1, self.vertex_decoder[-1].out_features // 3, 3)
        faces = faces.view(-1, self.face_decoder[-1].out_features // 3, 3)
        
        return vertices, faces

# === 3. Обучение ===
def train(model, dataset, device, epochs=100, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vertex_loss_fn = nn.MSELoss()
    
    print("\n" + "="*50)
    print("Начало обучения модели")
    print("="*50)
    print(f"Всего эпох: {epochs}")
    print(f"Размер датасета: {len(dataset)} моделей")
    print(f"Оптимизатор: Adam (lr={lr})")
    print(f"Устройство: {device}")
    print("="*50 + "\n")
    
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        
        print(f"\nЭпоха {epoch+1}/{epochs}")
        print("-"*30)
        
        progress_bar = tqdm(dataset, desc=f"Обучение", 
                          bar_format='{l_bar}{bar:30}{r_bar}')
        
        for i, sample in enumerate(progress_bar):
            text = sample['name']
            vertices_gt = torch.tensor(sample['vertices'], dtype=torch.float32, device=device).unsqueeze(0)
            num_vertices = sample['num_vertices']
            
            # Прямой проход
            pred_vertices, _ = model(text)
            pred_vertices = pred_vertices.to(device)  # Убеждаемся, что предсказания на нужном устройстве
            
            # Считаем loss только для реальных вершин
            loss = vertex_loss_fn(pred_vertices[:, :num_vertices], vertices_gt[:, :num_vertices])
            
            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Обновляем статистику
            epoch_loss += loss.item()
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'текст': text,
                'вершин': num_vertices
            })
        
        # Считаем средний loss за эпоху
        avg_epoch_loss = epoch_loss / len(dataset)
        
        # Обновляем лучший loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"\nНовый лучший loss: {best_loss:.6f}")
            
            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')
            print("Модель сохранена в 'best_model.pth'")
        
        print(f"\nРезультаты эпохи {epoch+1}:")
        print(f"Средний loss: {avg_epoch_loss:.6f}")
        print(f"Лучший loss: {best_loss:.6f}")
        print("-"*30)
    
    print("\n" + "="*50)
    print("Обучение завершено!")
    print(f"Лучший loss: {best_loss:.6f}")
    print("="*50 + "\n")

# === 4. Генерация и сохранение ===
def generate_and_save(model, text, num_vertices, num_faces, output_path):
    model.eval()
    with torch.no_grad():
        pred_vertices, pred_faces = model(text)
        pred_vertices = pred_vertices[0].cpu().numpy()
        pred_faces = pred_faces[0].cpu().numpy()
    
    # Используем только нужное количество вершин и граней
    vertices = pred_vertices[:num_vertices]
    faces = pred_faces[:num_faces].astype(int)
    
    # Создаем и сохраняем модель
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_path)
    print(f"3D модель сохранена в {output_path}")

# === 5. Основная функция ===
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*50)
    print(f"Используется устройство: {device}")
    print("="*50)
    
    dataset_dir = Path('dataset')
    print("\nЗагрузка датасета...")
    dataset, max_vertices, max_faces = load_dataset(dataset_dir)
    
    if not dataset:
        print("\nОШИБКА: Датасет пуст или не найдено подходящих .obj файлов!")
        return
    
    print("\n" + "="*50)
    print("Информация о датасете:")
    print(f"Количество моделей: {len(dataset)}")
    print(f"Максимальное количество вершин: {max_vertices}")
    print(f"Максимальное количество граней: {max_faces}")
    print("="*50)
    
    print("\nИнициализация модели...")
    model = TextTo3DModel(max_vertices, max_faces).to(device)
    
    # Выводим информацию о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("Информация о модели:")
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print("="*50 + "\n")
    
    print("Начало обучения...")
    train(model, dataset, device, epochs=100, lr=1e-3)
    
    print("\nГенерация моделей...")
    output_dir = Path('generated_models')
    output_dir.mkdir(exist_ok=True)
    
    # Создаем прогресс-бар для генерации
    for sample in tqdm(dataset, desc="Генерация моделей"):
        text = sample['name']
        num_vertices = sample['num_vertices']
        num_faces = sample['num_faces']
        output_path = output_dir / f"generated_{text}.obj"
        generate_and_save(model, text, num_vertices, num_faces, output_path)
    
    print("\n" + "="*50)
    print("Все модели сгенерированы!")
    print(f"Результаты сохранены в папке: {output_dir}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 