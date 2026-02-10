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
        weights = self._constrained_normalize(weights, min_weight=0.05, max_weight=0.45)
        
        # Шаг 3: Округляем до 2 знаков с сохранением суммы = 1.0
        weights = self._round_preserve_sum(weights, decimals=2)
        
        return weights
    
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
        """
        n = len(weights)
        device = weights.device
        
        # Проверка математической выполнимости ограничений
        if n < 3:  # 1 / 0.45 ≈ 2.22 → нужно минимум 3 критерия
            return torch.ones(n, device=device) / n
        
        if n > 20:  # 1 / 0.05 = 20 → максимум 20 критериев
            return torch.ones(n, device=device) / n
        
        # Итеративная проекция
        weights = weights.clone()
        
        for _ in range(max_iterations):
            # 1. Обрезаем значения за пределы диапазона
            weights = torch.clamp(weights, min_weight, max_weight)
            
            # 2. Перенормируем сумму на 1
            total = weights.sum()
            if total == 0:
                weights = torch.ones(n, device=device) / n
                break
            
            weights = weights / total
            
            # 3. Проверяем сходимость
            if (weights >= min_weight - tolerance).all() and (weights <= max_weight + tolerance).all():
                break
        
        # Финальная обрезка и перенормировка
        weights = torch.clamp(weights, min_weight, max_weight)
        weights = weights / weights.sum()
        
        return weights
    
    def _round_preserve_sum(self, weights: torch.Tensor, decimals: int = 2) -> torch.Tensor:
        """
        Округление весов до указанного числа знаков с сохранением суммы = 1.0.
        
        Алгоритм "распределения остатка":
        1. Умножаем на 10^decimals и получаем целые части
        2. Вычисляем недостающие единицы до 100 (для 2 знаков)
        3. Распределяем недостающие единицы по элементам с наибольшими дробными остатками
        """
        factor = 10 ** decimals
        n = len(weights)
        
        # Умножаем и получаем целые части
        scaled = weights * factor
        integer_parts = torch.floor(scaled).long()
        fractional_parts = scaled - integer_parts.float()
        
        # Сумма целых частей
        current_sum = integer_parts.sum().item()
        target_sum = factor  # 100 для 2 знаков
        
        # Сколько единиц нужно добавить/убрать
        diff = int(target_sum - current_sum)
        
        # Создаем копию для модификации
        result = integer_parts.clone().float()
        
        if diff > 0:
            # Сортируем индексы по убыванию дробной части
            _, indices = torch.sort(fractional_parts, descending=True)
            for i in range(min(diff, n)):
                result[indices[i]] += 1
        elif diff < 0:
            # Сортируем по возрастанию дробной части
            _, indices = torch.sort(fractional_parts)
            for i in range(min(-diff, n)):
                result[indices[i]] -= 1
        
        # Преобразуем обратно в веса
        rounded = result / factor
        
        # Финальная проверка суммы (защита от численных ошибок)
        total = rounded.sum()
        if abs(total - 1.0) > 1e-5:
            # Коррекция последнего элемента
            rounded[-1] += (1.0 - total)
            # Повторное округление
            rounded = torch.round(rounded * factor) / factor
        
        return rounded