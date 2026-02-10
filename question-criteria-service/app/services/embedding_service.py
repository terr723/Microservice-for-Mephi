import torch
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str, device: str = "cpu"):
        print(f"Загрузка модели {model_name} на устройстве {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        print("✓ Модель загружена")

    def encode_question(self, question: str, topic: str | None = None) -> torch.Tensor:
        text = f"query: {topic} {question}" if topic else f"query: {question}"
        return self.model.encode(text, convert_to_tensor=True, device=self.model.device)

    def encode_criteria(self, criteria: List[str]) -> torch.Tensor:
        texts = [f"passage: {c}" for c in criteria]
        return self.model.encode(texts, convert_to_tensor=True, device=self.model.device)