import torch
import torch.nn as nn
import numpy as np
import trimesh
import os
from tqdm import tqdm
import logging
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        # Усредняем по последовательности
        pooled = embedded.mean(dim=1)  # [batch_size, embedding_dim]
        return self.encoder(pooled)  # [batch_size, hidden_dim]

class TextToMeshGenerator:
    def __init__(self, 
                 vocab_size=1000,
                 embedding_dim=256,
                 hidden_dim=512,
                 mesh_size=256,
                 device=None):
        """
        Инициализация генератора 3D моделей из текста.
        
        Args:
            vocab_size (int): Размер словаря
            embedding_dim (int): Размерность эмбеддингов
            hidden_dim (int): Размер скрытого слоя
            mesh_size (int): Размер сетки для генерации
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используется устройство: {self.device}")
        
        # Создаем простой текстовый энкодер
        self.text_encoder = SimpleTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Создаем декодер для генерации меша
        self.mesh_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, mesh_size * 3),  # 3 координаты для каждой вершины
        ).to(self.device)
        
        # Инициализация базовой сетки (сфера)
        self.base_mesh = self._create_base_mesh(mesh_size)
        self.mesh_size = mesh_size
        
        # Создаем простой словарь
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = vocab_size
        
    def _create_base_mesh(self, size):
        """Создает базовую сетку в форме сферы."""
        # Создаем сферу с помощью trimesh
        sphere = trimesh.creation.icosphere(subdivisions=2)
        return sphere
    
    def _text_to_tensor(self, text):
        """Преобразует текст в тензор индексов."""
        # Разбиваем текст на слова
        words = re.findall(r'\w+', text.lower())
        
        # Создаем индексы для новых слов
        for word in words:
            if word not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        # Преобразуем слова в индексы
        indices = [self.word_to_idx.get(word, 0) for word in words]
        
        # Если последовательность слишком длинная, обрезаем
        if len(indices) > 50:
            indices = indices[:50]
        # Если слишком короткая, дополняем нулями
        elif len(indices) < 50:
            indices = indices + [0] * (50 - len(indices))
            
        return torch.tensor(indices, device=self.device).unsqueeze(0)
    
    def _encode_text(self, text):
        """Кодирует текст в векторное представление."""
        with torch.no_grad():
            # Преобразуем текст в тензор
            text_tensor = self._text_to_tensor(text)
            # Получаем эмбеддинги
            return self.text_encoder(text_tensor)
    
    def _generate_mesh(self, text_embedding):
        """Генерирует меш на основе текстового представления."""
        # Генерируем смещения вершин
        with torch.no_grad():
            vertex_offsets = self.mesh_decoder(text_embedding)
            vertex_offsets = vertex_offsets.view(-1, 3).cpu().numpy()
        
        # Получаем базовые вершины
        base_vertices = np.array(self.base_mesh.vertices)
        
        # Применяем смещения к базовой сетке
        new_vertices = base_vertices + vertex_offsets[:len(base_vertices)]
        
        # Создаем новый меш
        new_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=self.base_mesh.faces
        )
        
        return new_mesh
    
    def _optimize_mesh(self, mesh, num_iterations=50):
        """Оптимизирует меш для улучшения качества."""
        # Копируем меш для оптимизации
        optimized_mesh = mesh.copy()
        
        for _ in tqdm(range(num_iterations), desc="Оптимизация меша"):
            # Сглаживание вершин
            optimized_mesh = optimized_mesh.smoothed()
            
            # Удаляем вырожденные грани
            optimized_mesh.process(validate=True)
            
            # Исправляем нормали
            optimized_mesh.fix_normals()
            
            # Удаляем дубликаты вершин
            optimized_mesh.process(validate=True)
        
        return optimized_mesh
    
    def _validate_mesh(self, mesh):
        """Проверяет валидность меша."""
        # Проверяем, что меш водонепроницаемый
        if not mesh.is_watertight:
            logger.warning("Меш не является водонепроницаемым")
            # Пытаемся исправить
            mesh.fill_holes()
        
        # Проверяем ориентацию граней
        if not mesh.is_winding_consistent:
            logger.warning("Несогласованная ориентация граней")
            mesh.fix_normals()
        
        # Проверяем на вырожденные грани
        if mesh.is_empty:
            raise ValueError("Меш пустой")
        
        # Проверяем на дубликаты вершин
        if len(mesh.vertices) != len(np.unique(mesh.vertices, axis=0)):
            logger.warning("Обнаружены дубликаты вершин")
            mesh.process(validate=True)
        
        return mesh
    
    def generate(self, text, output_path, optimize=True, validate=True):
        """
        Генерирует 3D модель из текстового описания.
        
        Args:
            text (str): Текстовое описание модели
            output_path (str): Путь для сохранения модели
            optimize (bool): Нужно ли оптимизировать меш
            validate (bool): Нужно ли проверять валидность меша
            
        Returns:
            tuple: (путь к OBJ файлу, путь к STL файлу)
        """
        try:
            logger.info(f"Генерация модели для текста: {text}")
            
            # Кодируем текст
            text_embedding = self._encode_text(text)
            
            # Генерируем меш
            mesh = self._generate_mesh(text_embedding)
            
            # Проверяем валидность
            if validate:
                logger.info("Проверка валидности меша...")
                mesh = self._validate_mesh(mesh)
            
            # Оптимизируем меш если нужно
            if optimize:
                logger.info("Начало оптимизации меша...")
                mesh = self._optimize_mesh(mesh)
            
            # Создаем директорию если её нет
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохраняем в разных форматах
            obj_path = f"{output_path}.obj"
            stl_path = f"{output_path}.stl"
            
            # Сохраняем OBJ
            mesh.export(obj_path)
            logger.info(f"Модель сохранена в OBJ: {obj_path}")
            
            # Сохраняем STL
            mesh.export(stl_path)
            logger.info(f"Модель сохранена в STL: {stl_path}")
            
            # Визуализируем модель
            self._visualize_mesh(mesh, f"{output_path}_preview.png")
            
            return obj_path, stl_path
            
        except Exception as e:
            logger.error(f"Ошибка при генерации модели: {str(e)}")
            raise
    
    def _visualize_mesh(self, mesh, output_path):
        """Визуализирует меш и сохраняет превью."""
        try:
            # Создаем сцену
            scene = trimesh.Scene(mesh)
            
            # Настраиваем камеру
            camera = trimesh.scene.Camera(
                resolution=(800, 600),
                fov=(60, 60)
            )
            
            # Рендерим сцену
            png = scene.save_image(
                resolution=(800, 600),
                visible=True
            )
            
            # Сохраняем изображение
            with open(output_path, 'wb') as f:
                f.write(png)
            
            logger.info(f"Превью сохранено: {output_path}")
            
        except Exception as e:
            logger.warning(f"Не удалось создать превью: {str(e)}")

def create_generator(device=None):
    """
    Создает и возвращает генератор 3D моделей.
    
    Args:
        device (str): Устройство для вычислений ('cuda' или 'cpu')
        
    Returns:
        TextToMeshGenerator: Готовый к использованию генератор
    """
    generator = TextToMeshGenerator(device=device)
    return generator

def get_output_path():
    """Создает путь для сохранения модели на основе текущей даты и времени."""
    # Создаем директорию output если её нет
    os.makedirs("output", exist_ok=True)
    
    # Генерируем имя файла на основе текущего времени
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output/model_{timestamp}"

def interactive_mode():
    """Запускает интерактивный режим генерации моделей."""
    print("\n=== Генератор 3D моделей из текста ===")
    print("Введите 'выход' для завершения программы")
    print("Введите 'помощь' для получения справки")
    
    # Создаем генератор
    try:
        generator = create_generator()
        print("\nГенератор успешно инициализирован!")
    except Exception as e:
        print(f"\nОшибка при инициализации генератора: {str(e)}")
        return
    
    while True:
        # Получаем текст от пользователя
        print("\n" + "="*50)
        text = input("\nВведите описание модели (или команду): ").strip()
        
        # Проверяем команды
        if text.lower() == 'выход':
            print("\nЗавершение работы генератора...")
            break
        elif text.lower() == 'помощь':
            print("\nСправка по использованию:")
            print("1. Введите текстовое описание модели, которую хотите сгенерировать")
            print("   Например: 'Красный куб с закругленными краями'")
            print("2. Дождитесь завершения генерации")
            print("3. Модель будет сохранена в форматах OBJ и STL")
            print("4. Также будет создано превью модели в формате PNG")
            print("\nДоступные команды:")
            print("- 'выход': завершить работу программы")
            print("- 'помощь': показать эту справку")
            continue
        elif not text:
            print("\nПожалуйста, введите описание модели")
            continue
        
        try:
            # Генерируем путь для сохранения
            output_path = get_output_path()
            
            print(f"\nГенерация модели для описания: '{text}'")
            print("Это может занять некоторое время...")
            
            # Генерируем модель
            obj_path, stl_path = generator.generate(text, output_path)
            
            print("\nМодель успешно сгенерирована!")
            print(f"OBJ файл: {obj_path}")
            print(f"STL файл: {stl_path}")
            print(f"Превью: {output_path}_preview.png")
            
        except Exception as e:
            print(f"\nОшибка при генерации модели: {str(e)}")
            print("Попробуйте другое описание или обратитесь к справке (команда 'помощь')")

if __name__ == "__main__":
    try:
        # Проверяем аргументы командной строки
        import sys
        if len(sys.argv) > 1:
            # Если передан текст как аргумент, генерируем модель
            text = " ".join(sys.argv[1:])
            output_path = get_output_path()
            generator = create_generator()
            obj_path, stl_path = generator.generate(text, output_path)
            print(f"\nМодель успешно сгенерирована!")
            print(f"OBJ файл: {obj_path}")
            print(f"STL файл: {stl_path}")
            print(f"Превью: {output_path}_preview.png")
        else:
            # Иначе запускаем интерактивный режим
            interactive_mode()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\n\nНеожиданная ошибка: {str(e)}")
    finally:
        print("\nСпасибо за использование генератора 3D моделей!") 