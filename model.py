import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig, BertTokenizer
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import sys
import os
from config import ModelConfig  # Добавляем импорт ModelConfig

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TEXT_FEATURES,
    POINT_FEATURES,
    TOPOLOGY_FEATURES,
    VOXEL_SIZE,
    NUM_POINTS,
    NUM_GENERATION_STEPS,
    DEVICE,
    config  # Импортируем объект конфигурации
)

class TransformerEncoder(nn.Module):
    """Трансформерный энкодер для обработки последовательностей"""
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Входной тензор формы [batch_size, seq_len, d_model]
            mask: Маска внимания формы [batch_size, seq_len, seq_len]
        Returns:
            Тензор формы [batch_size, seq_len, d_model]
        """
        x = self.layer_norm(x)
        return self.transformer(x, mask)

class PointTransformer(nn.Module):
    """Трансформер для обработки облаков точек"""
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = None,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        config = None
    ):
        super().__init__()
        
        if config is not None:
            hidden_dim = config.point_features
        elif hidden_dim is None:
            hidden_dim = 256  # Значение по умолчанию
            
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = nn.Linear(3, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Тензор точек формы (batch_size, num_points, 3)
            
        Returns:
            Тензор признаков формы (batch_size, num_points, hidden_dim)
        """
        # Проекция входных точек
        x = self.input_projection(points)
        
        # Добавляем позиционное кодирование
        pos_encoding = self.position_encoding(points)
        x = x + pos_encoding
        
        # Применяем трансформер
        x = self.transformer(x)
        
        # Финальная проекция и нормализация
        x = self.output_projection(x)
        x = self.layer_norm(x)
        
        return x

class TextTo3DModel(nn.Module):
    """Модель для генерации 3D моделей из текста"""
    def __init__(self, config: ModelConfig):
        """Инициализация модели.
        
        Args:
            config: Конфигурация модели
        """
        super().__init__()
        self.config = config
        
        # Инициализируем BERT модель
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states=True,
            return_dict=True
        )
        
        # Замораживаем веса BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Проекционный слой для текстовых признаков
        self.text_projection = nn.Linear(
            self.bert.config.hidden_size,
            32  # Размерность текстовых признаков
        )
        
        # Трансформер для точек
        self.point_transformer = PointTransformer(
            input_dim=3,  # x, y, z
            hidden_dim=32,  # Размерность признаков точек
            config=config
        )
        
        # Вычисляем размерность входных признаков для декодера
        decoder_input_dim = 32 + 32  # text_features + point_features
        decoder_input_dim *= config.num_points  # Умножаем на количество точек
        
        # Декодер для генерации точек
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 1024),  # [B, decoder_input_dim] -> [B, 1024]
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * config.num_points),  # [B, 3*N]
            nn.Tanh()  # Нормализуем выход в диапазон [-1, 1]
        )
        
        # Сеть для генерации топологии
        self.topology_net = nn.Sequential(
            nn.Linear(config.point_features + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        # Инициализируем веса
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, text_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Прямой проход модели.
        
        Args:
            text_tokens: Словарь с токенизированным текстом
                - input_ids: [B, L]
                - attention_mask: [B, L]
                - token_type_ids: [B, L]
                
        Returns:
            Тензор с координатами точек [B, N, 3]
        """
        # Проверяем размерности входных тензоров
        batch_size = text_tokens['input_ids'].size(0)
        
        # Получаем текстовые признаки из BERT
        bert_output = self.bert(
            input_ids=text_tokens['input_ids'],
            attention_mask=text_tokens['attention_mask'],
            token_type_ids=text_tokens['token_type_ids']
        )
        
        text_features = bert_output.last_hidden_state[:, 0, :]  # [B, H]
        text_features = self.text_projection(text_features)  # [B, 32]
        
        # Инициализируем случайные точки
        points = torch.randn(
            batch_size,
            self.config.num_points,
            3,
            device=self.config.device
        )
        
        # Получаем признаки точек
        point_features = self.point_transformer(points)  # [B, N, 32]
        
        # Расширяем текстовые признаки для каждой точки
        text_features = text_features.unsqueeze(1).expand(-1, self.config.num_points, -1)  # [B, N, 32]
        
        # Объединяем признаки и выравниваем для декодера
        combined_features = torch.cat([point_features, text_features], dim=-1)  # [B, N, 64]
        combined_features = combined_features.reshape(batch_size, -1)  # [B, N*64]
        
        # Генерируем новые точки
        new_points = self.decoder(combined_features)  # [B, 3*N]
        new_points = new_points.view(batch_size, self.config.num_points, 3)  # [B, N, 3]
        
        return new_points
    
    def generate(self, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Генерация 3D модели из текста.
        
        Args:
            text: Список текстовых описаний
            
        Returns:
            Кортеж (points, topology):
            - points: Тензор сгенерированных точек [B, N, 3]
            - topology: Тензор топологии [B, N, 3]
        """
        self.eval()
        
        # Токенизируем текст
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_tokens = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors='pt'
        )
        
        # Перемещаем токены на нужное устройство
        text_tokens = {k: v.to(self.config.device) for k, v in text_tokens.items()}
        
        # Генерируем точки
        with torch.no_grad():
            points = self.forward(text_tokens)  # [B, N, 3]
            
            # Генерируем топологию
            point_features = self.point_transformer(points)  # [B, N, 32]
            topology_input = torch.cat([point_features, points], dim=-1)  # [B, N, 35]
            topology = self.topology_net(topology_input)  # [B, N, 3]
            
        return points, topology

    def load_state_dict(self, state_dict, strict=True):
        """Переопределяем метод загрузки весов для обработки несоответствий"""
        # Создаем копию словаря состояний
        new_state_dict = state_dict.copy()
        
        # Определяем префикс для весов BERT
        bert_prefix = 'text_encoder.bert.' if any(k.startswith('text_encoder.bert.') for k in new_state_dict.keys()) else 'text_encoder.'
        
        # Загружаем веса BERT
        bert_state_dict = {k[len(bert_prefix):]: v for k, v in new_state_dict.items() if k.startswith(bert_prefix)}
        if bert_state_dict:
            # Загружаем веса эмбеддингов отдельно
            if 'embeddings.word_embeddings.weight' in bert_state_dict:
                saved_weights = bert_state_dict['embeddings.word_embeddings.weight']
                current_weights = self.bert.embeddings.word_embeddings.weight
                # Копируем только общие веса
                min_vocab_size = min(saved_weights.size(0), current_weights.size(0))
                current_weights.data[:min_vocab_size] = saved_weights[:min_vocab_size]
                del bert_state_dict['embeddings.word_embeddings.weight']
            
            # Загружаем остальные веса BERT
            self.bert.load_state_dict(bert_state_dict, strict=False)
            
            # Удаляем загруженные веса из основного словаря
            for k in bert_state_dict.keys():
                new_state_dict.pop(bert_prefix + k, None)
        
        # Загружаем остальные веса модели
        super().load_state_dict(new_state_dict, strict=False)
        
        return self

def create_model() -> TextTo3DModel:
    """Создает экземпляр модели с текущей конфигурацией"""
    try:
        model = TextTo3DModel(config)
        model = model.to(config.device)  # Используем device из конфигурации
        return model
    except Exception as e:
        raise RuntimeError(f"Ошибка при создании модели: {str(e)}")

def load_model(checkpoint_path, config=None):
    """Загружает модель из чекпоинта"""
    try:
        # Загружаем состояние модели
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Определяем размер словаря из сохраненного состояния
        bert_prefix = 'text_encoder.bert.' if any(k.startswith('text_encoder.bert.') for k in state_dict.keys()) else 'text_encoder.'
        emb_key = f'{bert_prefix}embeddings.word_embeddings.weight'
        if emb_key in state_dict:
            vocab_size = state_dict[emb_key].size(0)
            print(f"Определен размер словаря из чекпоинта: {vocab_size}")
        else:
            vocab_size = 30522
            print("Используется стандартный размер словаря BERT: 30522")
        
        # Создаем конфигурацию с правильным размером словаря
        if config is None:
            config = ModelConfig()
        
        # Создаем модель
        model = TextTo3DModel(config)
        
        # Загружаем веса
        model.load_state_dict(state_dict, strict=False)
        
        # Переносим модель на нужное устройство
        model = model.to(config.device)
        model.eval()
        
        print(f"Модель успешно загружена из {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        print("Проверьте наличие файла модели и его совместимость.")
        return None 