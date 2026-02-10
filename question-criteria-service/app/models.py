from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"

class NormalizationMethod(str, Enum):
    SOFTMAX = "softmax"
    MINMAX = "minmax"

class QuestionCriteriaRequest(BaseModel):
    question: str = Field(..., min_length=1)
    criteria: List[str] = Field(..., min_items=1, max_items=50)
    topic: Optional[str] = None
    metric: SimilarityMetric = SimilarityMetric.COSINE
    normalization: NormalizationMethod = NormalizationMethod.SOFTMAX

class CriteriaWeight(BaseModel):
    criterion: str
    weight: float
    similarity_score: float
    rank: int

class WeightResponse(BaseModel):
    question: str
    topic: Optional[str]
    total_criteria: int
    weights: List[CriteriaWeight]
    metric_used: str
    normalization_method: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)