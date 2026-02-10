from fastapi import APIRouter, HTTPException, Depends 
from typing import List
import time
import torch
from app.models import QuestionCriteriaRequest, WeightResponse, CriteriaWeight
from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService
from app.services.normalization_service import NormalizationService
from app.dependencies import (
    get_embedding_service,
    get_similarity_service,
    get_normalization_service
)

router = APIRouter()

@router.post("/calculate-weights", response_model=WeightResponse)
async def calculate_criteria_weights(
    request: QuestionCriteriaRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    similarity_service: SimilarityService = Depends(get_similarity_service),
    normalization_service: NormalizationService = Depends(get_normalization_service)
):
    start = time.time()
    
    try:
        # 1. Эмбеддинги
        q_emb = embedding_service.encode_question(request.question, request.topic)
        c_embs = embedding_service.encode_criteria(request.criteria)
        
        # 2. Сходство
        sims = similarity_service.calculate_similarity(q_emb, c_embs, request.metric.value)
        
        # 3. Нормализация
        weights = normalization_service.normalize(sims, request.normalization.value)
        
        # 4. Сортировка и формирование ответа
        sorted_idx = torch.argsort(weights, descending=True)
        criteria_weights = [
            CriteriaWeight(
                criterion=request.criteria[i],
                weight=weights[i].item(),
                similarity_score=sims[i].item(),
                rank=rank + 1
            )
            for rank, i in enumerate(sorted_idx)
        ]
        
        return WeightResponse(
            question=request.question,
            topic=request.topic,
            total_criteria=len(request.criteria),
            weights=criteria_weights,
            metric_used=request.metric.value,
            normalization_method=request.normalization.value,
            processing_time_ms=(time.time() - start) * 1000
        )
        
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "question-criteria-weight-calculator"}