import re
from fastapi import FastAPI, HTTPException, Depends, APIRouter, status
from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, UTC
from fastapi.middleware.cors import CORSMiddleware
import sqlalchemy
from sqlalchemy import func
import os
import logging
from huggingface_hub import HfApi, hf_hub_url
import requests
from functools import lru_cache
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks
from fastapi.responses import RedirectResponse

from database import engine, get_session
from sqlmodel import Field, Session, SQLModel, create_engine, select
from auth import get_current_user, get_optional_user
from hf_sync import run_sync, sync_single_model_version
from config import get_settings
from generator import run_generator_for_version, process_all_unconfigured, retry_unsupported_segmentation
from schema import (
    MLModelDB,
    MLModelRead,
    MLModelCreate,
    ModelVersionDB,
    ModelVersionRead,
    ModelVersionCreate,
    ModelVersionUpdate,
    UserDB,
    UserRead,
    UserBase, # Assuming UserBase is used for creation if you don't have UserCreate
    InferenceLogDB,
    InferenceLogCreate,
)

# get .env configs
settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Input DTO for the HF Import Request
class HFImportRequest(BaseModel):
    hf_id: str # e.g. "google/mobilenet_v2_1.0_224"


router = APIRouter()

# ==========================================
# SCHEDULER SETUP
# ==========================================
# Initialize the background scheduler for HuggingFace model sync
scheduler = BackgroundScheduler()

def start_scheduler():
    """Start the background scheduler when FastAPI starts."""
    if not scheduler.running:
        # Schedule the HF sync job to run daily at 2 AM UTC
        scheduler.add_job(
            run_sync,
            CronTrigger(hour=2, minute=0),  # 2 AM UTC daily
            id='hf_litert_sync',
            name='HuggingFace LiteRT Model Sync',
            replace_existing=True,
            misfire_grace_time=600,  # Allow up to 10 minutes for missed executions
        )
        scheduler.start()
        logger.info("Background scheduler started - HF LiteRT sync scheduled for 02:00 UTC daily")

def stop_scheduler():
    """Stop the scheduler when FastAPI shuts down."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler shut down")

# 1. Cache the valid tasks from HF (Refresh once per day/hour)
@lru_cache(maxsize=1)
def get_valid_hf_tasks():
    try:
        resp = requests.get("https://huggingface.co/api/tasks")
        if resp.status_code == 200:
            # Returns a dict where keys are task IDs
            return set(resp.json().keys())
    except:
        pass
    return set() # Fallback

# ==========================================
# 0. USERS API
# ==========================================

@router.get("/users/me", response_model=UserRead, tags=["Users"])
def get_current_user_profile(
    current_user: UserDB = Depends(get_current_user)
):
    """
    Fetch the profile of the currently authenticated user.
    """
    return current_user

@router.get("/users/{user_id}", response_model=UserRead, tags=["Users"])
def get_user(user_id: uuid.UUID, session: Session = Depends(get_session),
             current_user: UserDB = Depends(get_current_user)):
    """Fetch a public user profile."""
    user = session.get(UserDB, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED, tags=["Users"])
def create_user(user_in: UserBase, session: Session = Depends(get_session),
                current_user: UserDB = Depends(get_current_user)):
    """
    Create a new user. 
    Typically called after a successful Supabase signup webhook or first login.
    """
    # Check for existing user
    existing_user = session.exec(select(UserDB).where(UserDB.email == user_in.email)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    new_user = UserDB.model_validate(user_in)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user

@router.get("/users/{user_id}/models", response_model=List[MLModelRead], tags=["Users"])
def get_user_models(
    user_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 50, 
    session: Session = Depends(get_session),
    current_user: UserDB = Depends(get_current_user)
):
    """
    Convenience route: Fetch all top-level models authored by a specific user.
    """
    # Verify user exists
    user = session.get(UserDB, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    query = select(MLModelDB).where(MLModelDB.author_id == user_id).offset(skip).limit(limit)
    models = session.exec(query).all()
    return models

# ==========================================
# 1. MODELS API
# ==========================================

@router.get("/models", response_model=List[MLModelRead], tags=["Models"])
def get_all_models(
    skip: int = 0,
    limit: int = 1000,
    task: str = None,
    supported_only: bool = False,
    author_id: Optional[uuid.UUID] = None,
    session: Session = Depends(get_session),
    current_user: Optional[UserDB] = Depends(get_optional_user),
):
    """
    Fetch the high-level model summaries.
    We use query parameters for filtering, adhering to strict REST guidelines.
    """
    query = select(MLModelDB)

    # Visibility: public models are visible to all; private models only to their author
    if current_user:
        query = query.where(
            sqlalchemy.or_(MLModelDB.is_public == True, MLModelDB.author_id == current_user.id)
        )
    else:
        query = query.where(MLModelDB.is_public == True)

    if author_id:
        query = query.where(MLModelDB.author_id == author_id)

    if task:
        query = query.where(MLModelDB.task == task)

    if supported_only:
        query = query.join(ModelVersionDB).where(ModelVersionDB.status.in_(["supported", "unverified"])).distinct()

    query = query.offset(skip).limit(limit)
    models = session.exec(query).all()

    # Fetch version counts and max file sizes in single GROUP BY queries instead of N+1 loads
    version_counts = dict(
        session.exec(
            select(ModelVersionDB.model_id, func.count(ModelVersionDB.id))
            .group_by(ModelVersionDB.model_id)
        ).all()
    )
    file_sizes = dict(
        session.exec(
            select(ModelVersionDB.model_id, func.max(ModelVersionDB.file_size_bytes))
            .group_by(ModelVersionDB.model_id)
        ).all()
    )

    # Compute the "best" version status per model (supported > configured > unverified > unconfigured > broken > unsupported)
    _STATUS_PRIORITY = {"supported": 0, "configured": 1, "unverified": 2, "unconfigured": 3, "broken": 4, "unsupported": 5}
    best_statuses: dict = {}
    for vid, vstatus in session.exec(select(ModelVersionDB.model_id, ModelVersionDB.status)).all():
        current = best_statuses.get(vid)
        if current is None or _STATUS_PRIORITY.get(vstatus, 99) < _STATUS_PRIORITY.get(current, 99):
            best_statuses[vid] = vstatus

    return [
        MLModelRead(
            id=m.id,
            author_id=m.author_id,
            name=m.name,
            slug=m.slug,
            description=m.description or "",
            category=m.category,
            tags=m.tags or [],
            task=m.task,
            hf_model_id=m.hf_model_id,
            is_verified_official=m.is_verified_official,
            is_public=m.is_public,
            total_download_count=m.total_download_count,
            total_ratings=m.total_ratings,
            rating_weighted_avg=m.rating_weighted_avg,
            created_at=m.created_at,
            version_count=version_counts.get(m.id, 0),
            file_size_bytes=file_sizes.get(m.id, 0) or 0,
            best_version_status=best_statuses.get(m.id),
        )
        for m in models
    ]

@router.get("/models/{model_id}", response_model=MLModelRead, tags=["Models"])
def get_model(
    model_id: uuid.UUID,
    session: Session = Depends(get_session),
    current_user: Optional[UserDB] = Depends(get_optional_user),
):
    """Fetch a specific model summary."""
    model = session.get(MLModelDB, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.is_public:
        if not current_user or current_user.id != model.author_id:
            raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.post("/models", response_model=MLModelRead, status_code=status.HTTP_201_CREATED, tags=["Models"])
def create_model(
    model_in: MLModelCreate,
    session: Session = Depends(get_session),
    current_user: UserDB = Depends(get_current_user)
):
    """Create a new top-level model resource."""
    author_id = current_user.id

    # Auto-generate a unique slug from the model name
    base_slug = re.sub(r'[^a-z0-9]+', '-', model_in.name.lower()).strip('-')
    slug = base_slug
    suffix = 1
    while session.exec(select(MLModelDB).where(MLModelDB.slug == slug)).first():
        slug = f"{base_slug}-{suffix}"
        suffix += 1

    new_model = MLModelDB(**model_in.model_dump(exclude={"slug"}), author_id=author_id, slug=slug)
    session.add(new_model)
    session.commit()
    session.refresh(new_model)
    return new_model


# ==========================================
# 2. MODEL VERSIONS API 
# ==========================================

@router.get("/models/{model_id}/versions", response_model=List[ModelVersionRead], tags=["Model Versions"])
def get_model_versions(model_id: uuid.UUID, 
                       session: Session = Depends(get_session)):
    """
    Fetch all versions for a specific model.
    The response automatically maps the JSONB 'assets' column to the Pydantic AssetPointers model.
    """
    # Verify model exists
    model = session.get(MLModelDB, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    versions = session.exec(
        select(ModelVersionDB).where(ModelVersionDB.model_id == model_id)
    ).all()
    
    return versions

@router.post("/models/{model_id}/versions", response_model=ModelVersionRead, status_code=status.HTTP_201_CREATED, tags=["Model Versions"])
def create_model_version(
    model_id: uuid.UUID,
    version_in: ModelVersionCreate,
    session: Session = Depends(get_session),
    current_user: UserDB = Depends(get_current_user),
):
    """Manually create a new model version. Provide pipeline_spec to set status to 'configured'."""
    model = session.get(MLModelDB, model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    initial_status = "configured" if version_in.pipeline_spec else "unconfigured"
    new_version = ModelVersionDB(
        model_id=model_id,
        version_name=version_in.version_name,
        commit_sha=version_in.commit_sha,
        assets=version_in.assets.model_dump(mode='json'),
        license_type=version_in.license_type,
        is_commercial_safe=version_in.is_commercial_safe,
        requires_commercial_warning=version_in.requires_commercial_warning,
        file_size_bytes=version_in.file_size_bytes,
        status=initial_status,
        changelog=version_in.changelog,
        pipeline_spec=version_in.pipeline_spec.model_dump(mode='json') if version_in.pipeline_spec else None,
        pipeline_updated_at=datetime.now(UTC) if version_in.pipeline_spec else None,
    )
    session.add(new_version)
    session.commit()
    session.refresh(new_version)
    return new_version

@router.get("/versions/{version_id}", response_model=ModelVersionRead, tags=["Model Versions"])
def get_version(version_id: uuid.UUID, session: Session = Depends(get_session)):
    """Fetch a specific version and its dynamic asset pointers."""
    version = session.get(ModelVersionDB, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    return version

@router.patch("/versions/{version_id}", response_model=ModelVersionRead, tags=["Model Versions"])
def update_model_version_config(
    version_id: uuid.UUID,
    payload: ModelVersionUpdate,
    session: Session = Depends(get_session),
    current_user: UserDB = Depends(get_current_user)
):
    """
    Pocket AI Strategy 3: Crowdsourced Configuration Ingestion.
    Accepts a rigorously validated pipeline_spec from the React UI and commits it.
    """
    # 1. Fetch the Target Version
    version = session.get(ModelVersionDB, version_id)
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Model version not found."
        )

    # 2. Safely Update Pipeline Spec (Only if provided in the PATCH payload)
    if payload.pipeline_spec is not None:
        # Pydantic safely converts the complex PipelineConfig object to a basic dict for Postgres
        version.pipeline_spec = payload.pipeline_spec.model_dump(mode='json')
        version.pipeline_updated_at = datetime.now(UTC)

    # 3. Safely Update Status (Only if provided)
    if payload.status is not None:
        version.status = payload.status

    # 4. Commit to Postgres
    session.add(version)
    session.commit()
    session.refresh(version)

    return version

@router.delete("/versions/{version_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Model Versions"])
def delete_model_version(
    version_id: uuid.UUID,
    session: Session = Depends(get_session),
    current_user: UserDB = Depends(get_current_user),
):
    """Permanently delete a model version and all its associated data."""
    version = session.get(ModelVersionDB, version_id)
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found.")
    session.delete(version)
    session.commit()

@router.get("/versions/{version_id}/download/{asset_key}", tags=["Model Versions", "Downloads"])
def download_model_asset(
    version_id: uuid.UUID,
    asset_key: str,
    session: Session = Depends(get_session)
):
    """
    Download a specific asset (e.g., 'tflite', 'labels') for a model version.
    This endpoint resolves the asset's storage location and redirects the client,
    allowing us to track telemetry and abstract the underlying storage provider.
    """
    # 1. Fetch the Target Version
    version = session.get(ModelVersionDB, version_id)
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Model version not found."
        )

    # 2. Check if the requested asset exists
    asset_url = version.assets.get(asset_key)
    if not asset_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Asset '{asset_key}' not found for this version."
        )

    # 3. Telemetry: Increment download counts 
    # We only increment if they download the main binary, not just a label file.
    if asset_key == "tflite":
        version.download_count += 1
        session.add(version)
        
        # Also increment parent model download count
        model = session.get(MLModelDB, version.model_id)
        if model:
            model.total_download_count += 1
            session.add(model)
            
        session.commit()

    # 4. Redirect the client directly to the storage provider (Hugging Face, S3, etc.)
    # Using a 307 Temporary Redirect ensures the client follows the redirect to fetch the file
    # but doesn't permanently cache the resolution, so you can change storage providers later.
    return RedirectResponse(url=asset_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)

@router.delete("/versions/{version_id}/pipeline", response_model=ModelVersionRead, tags=["Model Versions"])
def delete_pipeline_config(
    version_id: uuid.UUID,
    session: Session = Depends(get_session),
    current_user: UserDB = Depends(get_current_user)
):
    """
    Clears the pipeline configuration for a model version, reverting it to unconfigured.
    """
    version = session.get(ModelVersionDB, version_id)
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found.")

    version.pipeline_spec = None
    version.status = "unconfigured"
    version.unsupported_reason = None
    version.pipeline_updated_at = None

    session.add(version)
    session.commit()
    session.refresh(version)
    return version


# generate a pipeline config for a specific model version
@router.post("/versions/{version_id}/generate-pipeline", response_model=ModelVersionRead, tags=["Model Versions"])
def trigger_pipeline_generation(
    version_id: uuid.UUID,
    session: Session = Depends(get_session),
    current_user: UserDB = Depends(get_current_user)
):
    """
    Manually triggers the LLM to generate a pipeline config for a specific model version.
    """
    # 1. Fetch the Target Version and Parent Model
    version = session.get(ModelVersionDB, version_id)
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Model version not found."
        )
        
    model = session.get(MLModelDB, version.model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Parent model not found."
        )

    # 2. Call the generator helper
    success = run_generator_for_version(version, model, session)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM generation failed. Please check the server logs."
        )
        
    # 3. Refresh and return the updated version
    session.refresh(version)
    return version

@router.post("/versions/generate-pipeline-all", tags=["Model Versions"])
def trigger_pipeline_generation_all(
    background_tasks: BackgroundTasks,
    current_user: UserDB = Depends(get_current_user)
):
    """
    Kicks off a background job to scan the database for all 'unconfigured'
    model versions and generate their pipelines using the LLM.
    """
    # FastAPI will pass this function to a background worker and return the HTTP response immediately
    background_tasks.add_task(process_all_unconfigured)

    return {
        "message": "Batch pipeline generation started in the background. Check server logs for progress.",
        "status": "processing"
    }

@router.post("/versions/retry-segmentation", tags=["Model Versions"])
def retry_segmentation_models(
    background_tasks: BackgroundTasks,
    current_user: UserDB = Depends(get_current_user)
):
    """
    Re-runs pipeline generation for all model versions that were previously
    rejected and belong to a segmentation-task model. Use this after updating
    the generator rules to support semantic segmentation.
    """
    background_tasks.add_task(retry_unsupported_segmentation)
    return {
        "message": "Segmentation model retry started in the background. Check server logs for progress.",
        "status": "processing"
    }


# Search HF for models with TFLite files
class HFSearchResult(BaseModel):
    id: str
    description: str
    tags: List[str]
    pipeline_tag: Optional[str] = None

class HFSearchResponse(BaseModel):
    results: List[HFSearchResult]

@router.get("/search/huggingface", response_model=HFSearchResponse, tags=["Hugging Face"], summary="Search Hugging Face for TFLite models")
def search_huggingface(query: str, 
                       ):
    """
    Search Hugging Face for models that contain .tflite files.
    Filters to only return models with actual TFLite files.
    """
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    try:
        # Use HF token if available from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            api = HfApi(token=hf_token) 
            print("Searching HuggingFace using token")
        else:
            api = HfApi()
            print("Searching HuggingFace without token")
        
        # Search HF for models without framework filter to avoid auth issues
        # The pytorch filter was too restrictive and causing rate limit issues
        hf_models = api.list_models(
            search=query,
            limit=30,
            full=False  # Don't fetch full metadata for initial search
        )
        
        results = []
        
        # Filter to only models with .tflite files
        for model_info in hf_models:
            try:
                # Get detailed info with file listing
                detailed_info = api.model_info(repo_id=model_info.id, files_metadata=True)
                
                # Check if it has tflite files
                if detailed_info.siblings is None:
                    continue
                    
                tflite_files = [f for f in detailed_info.siblings if f.rfilename.endswith(".tflite")]
                
                if tflite_files:  # Only include if it has TFLite files
                    # Safely extract description
                    description = ""
                    if detailed_info.cardData and isinstance(detailed_info.cardData, dict):
                        description = detailed_info.cardData.get("summary", "") or detailed_info.cardData.get("description", "")
                    
                    results.append(HFSearchResult(
                        id=model_info.id,
                        description=description if description else "TFLite model from Hugging Face",
                        tags=detailed_info.tags or [],
                        pipeline_tag=detailed_info.pipeline_tag
                    ))
                    
                    # Limit results to avoid too many API calls
                    if len(results) >= 15:
                        break
                        
            except Exception as e:
                # Log but skip models that fail to load details
                # Common issues: private models, API timeouts, etc.
                continue
        
        return HFSearchResponse(results=results)
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Unable to connect to Hugging Face. Please check your internet connection and try again.")
    except Exception as e:
        error_msg = str(e)
        # Be more specific about common errors
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise HTTPException(status_code=401, detail="Hugging Face API error: Please check if the HF token is valid or if you've hit the rate limit. Try again later.")
        elif "rate" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Hugging Face rate limit reached. Please wait a moment and try again.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to search Hugging Face: {error_msg[:100]}")

# ==========================================
# HUGGINGFACE LITERT SYNC ENDPOINT
# ==========================================
class SyncResponse(BaseModel):
    status: str
    created: int
    updated: int
    skipped: int
    message: str

@router.post("/sync/huggingface/litert", response_model=SyncResponse, tags=["Hugging Face"], summary="Manually trigger HuggingFace LiteRT model sync")
def manual_sync_literrt_models(current_user: UserDB = Depends(get_current_user)):
    """
    Manually trigger the HuggingFace LiteRT model sync job.
    This endpoint is protected - only authenticated users can trigger it.
    Useful for testing or force-syncing models outside the scheduled time.
    """
    # In production, you might want to restrict this to admin users
    try:
        logger.info(f"Manual sync triggered by user: {current_user.username}")
        stats = run_sync(limit=settings.HF_SYNC_FETCH_LIMIT)
        
        return SyncResponse(
            status="success",
            created=stats.get("created", 0),
            updated=stats.get("updated", 0),
            skipped=stats.get("skipped", 0),
            message=f"Successfully synced LiteRT models. Created: {stats.get('created', 0)}, Updated: {stats.get('updated', 0)}, Skipped: {stats.get('skipped', 0)}"
        )
    except Exception as e:
        logger.error(f"Error during manual sync: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
    
@router.post("/sync/huggingface/litert/adhoc", tags=["Hugging Face"], summary="Manually trigger sync for a specific HF repo and commit")
def manual_sync_specific_version(
    repo_id: str, 
    commit_sha: str, 
    current_user: UserDB = Depends(get_current_user)
):
    """
    Manually trigger the sync process for a specific Hugging Face repository and commit.
    Useful for testing or force-syncing a single model version outside the scheduled job.
    """
    try:
        logger.info(f"Manual sync for repo: {repo_id}, commit: {commit_sha} triggered by user: {current_user.username}")
        sync_single_model_version(repo_id=repo_id, commit_sha=commit_sha)
        
        return {"status": "success", "message": f"Manual sync for {repo_id} at {commit_sha[:7]} completed. Check logs for details."}
        
    except Exception as e:
        logger.error(f"Error during manual sync for {repo_id} at {commit_sha}: {e}")
        raise HTTPException(status_code=500, detail=f"Manual sync failed: {str(e)}")


@router.post("/telemetry/batch", status_code=status.HTTP_201_CREATED, tags=["Telemetry"])
def upload_telemetry_batch(
    batch: List[InferenceLogCreate],
    current_user: UserDB = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Accept a batch of on-device inference stats from an authenticated user.
    Each entry is persisted to inference_logs for analytics.
    """
    if not batch:
        return {"accepted": 0}

    created = 0
    for entry in batch:
        # Verify the version exists before inserting
        version = session.get(ModelVersionDB, entry.model_version_id)
        if version is None:
            continue  # Skip unknown versions silently
        log = InferenceLogDB(
            model_version_id=entry.model_version_id,
            device_model=entry.device_model,
            platform=entry.platform,
            total_inference_ms=entry.total_inference_ms,
            success=entry.success,
            top_confidence=entry.top_confidence,
            num_results=entry.num_results,
            task_type=entry.task_type,
        )
        session.add(log)
        created += 1

    session.commit()
    logger.info(f"Telemetry batch: {created}/{len(batch)} records saved (user={current_user.id})")
    return {"accepted": created}