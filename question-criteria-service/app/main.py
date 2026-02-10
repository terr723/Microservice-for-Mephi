from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints import router
from app.dependencies import service_container, get_settings_dep
from app.config import Settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Старт: инициализация сервисов
    settings: Settings = get_settings_dep()
    print(f"\n🚀 Запуск сервиса в окружении: {settings.ENVIRONMENT}")
    print(f"   Модель: {settings.EMBEDDING_MODEL_NAME}")
    print(f"   Устройство: {settings.MODEL_DEVICE}\n")
    
    # Embedding сервис инициализируется при первом запросе через DI
    yield
    
    # Завершение
    print("\n🛑 Сервис остановлен\n")

app = FastAPI(
    title="Question Criteria Weight Calculator",
    description="Микросервис для расчета весов критериев через семантическое сходство",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "Question Criteria Weight Calculator",
        "docs": "/docs",
        "api": "/api/v1/calculate-weights"
    }