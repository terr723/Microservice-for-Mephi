import torch

class NormalizationService:
    def normalize(self, similarities: torch.Tensor, method: str = "softmax") -> torch.Tensor:
        method = method.lower()
        
        if method == "softmax":
            return torch.softmax(similarities, dim=0)
        
        elif method == "minmax":
            sim = similarities.cpu()
            if sim.max() == sim.min():
                return torch.ones_like(similarities) / len(similarities)
            norm = (sim - sim.min()) / (sim.max() - sim.min())
            return (norm / norm.sum()).to(similarities.device)
        
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")