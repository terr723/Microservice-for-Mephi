import torch
import torch.nn.functional as F

class SimilarityService:
    def calculate_similarity(self, question_emb: torch.Tensor, criteria_embs: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
        metric = metric.lower()
        
        if metric == "cosine":
            q = F.normalize(question_emb.unsqueeze(0), p=2, dim=1)
            c = F.normalize(criteria_embs, p=2, dim=1)
            return torch.matmul(q, c.T).squeeze(0)
        
        elif metric == "dot":
            return torch.matmul(question_emb.unsqueeze(0), criteria_embs.T).squeeze(0)
        
        elif metric == "euclidean":
            dist = torch.cdist(question_emb.unsqueeze(0), criteria_embs, p=2)
            return (1 / (1 + dist)).squeeze(0)
        
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")