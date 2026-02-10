import torch

class NormalizationService:
    def normalize(self, similarities: torch.Tensor, method: str = "softmax") -> torch.Tensor:
        method = method.lower()
        
        # Шаг 1: Применяем базовую нормализацию
        if method == "softmax":
            weights = torch.softmax(similarities, dim=0)
        elif method == "minmax":
            sim = similarities.cpu()
            if sim.max() == sim.min():
                weights = torch.ones_like(similarities) / len(similarities)
            else:
                norm = (sim - sim.min()) / (sim.max() - sim.min())
                weights = (norm / norm.sum()).to(similarities.device)
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")
        
        # Шаг 2: Применяем ограничения [0.05, 0.45] с итеративной проекцией
        return self._constrained_normalize(weights, min_weight=0.05, max_weight=0.45)
    
    def _constrained_normalize(
        self, 
        weights: torch.Tensor, 
        min_weight: float = 0.05, 
        max_weight: float = 0.45,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> torch.Tensor:
        """
        Итеративная проекция весов на симплекс с ограничениями [min_weight, max_weight].
        
        Алгоритм:
        1. Обрезаем значения за пределы диапазона
        2. Перенормируем сумму на 1
        3. Повторяем до сходимости или достижения макс. итераций
        
        Математическое ограничение:
        - Ограничение выполнимо только если: min_weight * n ≤ 1 ≤ max_weight * n
        - Для [0.05, 0.45]: 3 ≤ n ≤ 20 критериев
        """
        n = len(weights)
        device = weights.device
        
        # Проверка математической выполнимости ограничений
        if n < 3:  # 1 / 0.45 ≈ 2.22 → нужно минимум 3 критерия
            # Для 1-2 критериев используем равномерное распределение (ограничение невозможно)
            return torch.ones(n, device=device) / n
        
        if n > 20:  # 1 / 0.05 = 20 → максимум 20 критериев
            # Для >20 критериев используем равномерное распределение (ограничение невозможно)
            return torch.ones(n, device=device) / n
        
        # Итеративная проекция на ограниченный симплекс
        weights = weights.clone()
        
        for _ in range(max_iterations):
            # 1. Обрезаем значения за пределы диапазона
            weights = torch.clamp(weights, min_weight, max_weight)
            
            # 2. Перенормируем сумму на 1
            total = weights.sum()
            if total == 0:
                # Защита от деления на ноль
                weights = torch.ones(n, device=device) / n
                break
            
            weights = weights / total
            
            # 3. Проверяем сходимость (все веса в диапазоне с небольшим допуском)
            if (weights >= min_weight - tolerance).all() and (weights <= max_weight + tolerance).all():
                break
        
        # Финальная обрезка для устранения численных ошибок
        weights = torch.clamp(weights, min_weight, max_weight)
        
        # Финальная перенормировка (гарантируем сумму = 1)
        weights = weights / weights.sum()
        
        return weights