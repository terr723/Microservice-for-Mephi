# Экспортируем сервисы для удобного импорта
from .embedding_service import EmbeddingService
from .similarity_service import SimilarityService
from .normalization_service import NormalizationService

__all__ = [
    "EmbeddingService",
    "SimilarityService",
    "NormalizationService"
]