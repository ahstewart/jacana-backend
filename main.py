import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router, start_scheduler, stop_scheduler
from database import engine
from sqlmodel import SQLModel
from config import get_settings

# Define metadata for your Swagger documentation groups
tags_metadata = [
    {
        "name": "Users",
        "description": "Manage user profiles, registration, and dashboard data.",
    },
    {
        "name": "Models",
        "description": "Top-level ML models, discoverability, and category filtering.",
    },
    {
        "name": "Model Versions",
        "description": "Specific deployment binaries, configurations, and pipeline specs.",
    },
]

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Pocket AI Lab API",
    description="Backend service for managing LiteRT models and telemetry.",
    version="1.0.0",
    openapi_tags=tags_metadata
)

# 2. Configure CORS Middleware (Strictly Enforced Architecture Rule)
# This MUST be added before any routes to prevent 400 OPTIONS errors
# from masking underlying 500 server crashes during local development.
origins = [
    "http://localhost:5173",  # Default Vite React dev server port
    "http://127.0.0.1:5173",
    "https://gumbo-zeta.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow GET, POST, PATCH, etc.
    allow_headers=["*"],
)

# 3. Create Database Tables (Development Convenience)
# In production, you would use Alembic migrations instead of doing this on startup.
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)
    print("Database tables verified.")
    start_scheduler()

@app.on_event("shutdown")
def on_shutdown():
    stop_scheduler()

# 4. Mount the API Router
# We prefix all routes from api.py with /api/v1 for clean versioning
app.include_router(api_router, prefix="/api/v1")

# 5. Root Health Check Route
@app.get("/")
def read_root():
    return {"message": "Pocket AI Lab API is running.", "status": "ok"}

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)