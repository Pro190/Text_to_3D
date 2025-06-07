import torch
import torch.nn as nn
import torch.nn.functional as F

class ChamferDistance(nn.Module):
    """Функция потерь Chamfer Distance для сравнения облаков точек."""
    
    def __init__(self, reduction: str = 'mean'):
        """Инициализация функции потерь.
        
        Args:
            reduction: Способ редукции потерь ('mean' или 'sum')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Вычисляет расстояние Чамфера между предсказанным и целевым облаками точек.
        
        Args:
            pred: Предсказанное облако точек [B, N, 3]
            target: Целевое облако точек [B, M, 3]
            
        Returns:
            Расстояние Чамфера
        """
        # Нормализуем точки
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        
        # Вычисляем попарные расстояния между точками
        dist = torch.cdist(pred, target)  # [B, N, M]
        
        # Находим минимальные расстояния в обоих направлениях
        min_dist_pred = torch.min(dist, dim=2)[0]  # [B, N]
        min_dist_target = torch.min(dist, dim=1)[0]  # [B, M]
        
        # Суммируем расстояния
        chamfer_dist = min_dist_pred.sum(dim=1) + min_dist_target.sum(dim=1)  # [B]
        
        # Применяем редукцию
        if self.reduction == 'mean':
            return chamfer_dist.mean()
        elif self.reduction == 'sum':
            return chamfer_dist.sum()
        else:
            raise ValueError(f"Недопустимое значение reduction: {self.reduction}")
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Вызывает forward метод."""
        return self.forward(pred, target) 