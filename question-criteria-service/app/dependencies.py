from typing import Annotated
from fastapi import Depends
from app.config import get_settings, Settings
from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService
from app.services.normalization_service import NormalizationService

# Singleton контейнер
class ServiceContainer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.embedding_service = None
            cls._instance.similarity_service = SimilarityService()
            cls._instance.normalization_service = NormalizationService()
        return cls._instance

service_container = ServiceContainer()

# Factory функции для DI
def get_settings_dep() -> Settings:
    return get_settings()

def get_embedding_service(settings: Annotated[Settings, Depends(get_settings_dep)]):
    if service_container.embedding_service is None:
        service_container.embedding_service = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.MODEL_DEVICE
        )
    return service_container.embedding_service

def get_similarity_service():
    return service_container.similarity_service

def get_normalization_service():
    return service_container.normalization_service