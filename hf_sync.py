"""
HuggingFace LiteRT Model Syncer

This module fetches all public models with "LiteRT" library from HuggingFace
and creates/updates corresponding model objects in Jacana database.

This script is designed to run daily via APScheduler.
"""

import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
from sqlmodel import Session, select
from huggingface_hub import HfApi
import sqlalchemy
import time
import requests

from schema import MLModelDB, ModelVersionDB, UserDB, ModelCategory
from database import engine
from config import get_settings
from scanner import scan_hf_repo_for_version_assets

# get .env configs
settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_or_create_system_user(session: Session) -> uuid.UUID:
    """
    Ensures we have a 'Jacana System' user to act as the author
    for automatically synced open-source models.
    """
    system_email = "system@jacana.app"
    user = session.exec(select(UserDB).where(UserDB.email == system_email)).first()

    if not user:
        user = UserDB(
            username="JacanaSystem",
            email=system_email,
            is_developer=True,
            hf_username="jacana_system"
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        
    return user.id

def extract_tag_value(tags: list, prefix: str) -> str:
    """Helper to safely extract tags like 'license:mit' -> 'mit'."""
    if not tags:
        return ""
    for tag in tags:
        if tag.startswith(prefix):
            return tag.replace(prefix, "")
    return ""


def interpolate_model_task(hf_model) -> str:
    """
    Attempts to determine the task of a Hugging Face model.
    Accepts an object/dict representing the model data from the Hugging Face API.
    """
    # If using the huggingface_hub library, access attributes directly. 
    # If using a raw dict, change these to hf_model.get("...")
    pipeline_tag = getattr(hf_model, "pipeline_tag", None)
    model_id = getattr(hf_model, "modelId", "").lower()
    tags = [t.lower() for t in getattr(hf_model, "tags", [])]

    # ==========================================
    # TIER 1: The Explicit Tag
    # ==========================================
    if pipeline_tag:
        return pipeline_tag

    # ==========================================
    # TIER 2: Keyword Matching in Model ID
    # Many users name their models after the architecture or dataset.
    # ==========================================
    if "whisper" in model_id or "wav2vec" in model_id:
        return "automatic-speech-recognition"
    
    if "yolo" in model_id or "ssd" in model_id or "centernet" in model_id:
        return "object-detection"
    
    if "resnet" in model_id or "mobilenet" in model_id or "efficientnet" in model_id:
        return "image-classification"
        
    if "deeplab" in model_id or "segmentation" in model_id:
        return "image-segmentation"
        
    if "distilbert" in model_id or "roberta" in model_id or "sst2" in model_id:
        return "text-classification"

    # ==========================================
    # TIER 3: Tag Scanning
    # Sometimes users forget the pipeline_tag but add generic tags.
    # ==========================================
    for tag in tags:
        if "speech-recognition" in tag or "asr" in tag:
            return "automatic-speech-recognition"
        if "vision" in tag and "classification" in tag:
            return "image-classification"
        if "object-detection" in tag:
            return "object-detection"
        if "nlp" in tag and "classification" in tag:
            return "text-classification"

    # ==========================================
    # TIER 4: The Config Check
    # If it's a transformer model, it has a config.json file that lists its exact architecture.
    # ==========================================
    try:
        # We ping the raw config file directly from HF
        config_url = f"https://huggingface.co/{hf_model.modelId}/resolve/main/config.json"
        response = requests.get(config_url, timeout=3)
        
        if response.status_code == 200:
            config = response.json()
            architectures = config.get("architectures", [])
            
            if architectures:
                arch = architectures[0].lower()
                if "forconditionalgeneration" in arch and ("whisper" in arch or "speech" in arch):
                    return "automatic-speech-recognition"
                if "forsequenceclassification" in arch:
                    return "text-classification"
                if "forimageclassification" in arch:
                    return "image-classification"
                if "forobjectdetection" in arch:
                    return "object-detection"
    except Exception as e:
        # Fail gracefully so the nightly sync doesn't crash on network errors
        print(f"Failed to fetch config for {hf_model.modelId}: {e}")

    # ==========================================
    # TIER 5: Fallback
    # ==========================================
    return "unknown"

def sync_huggingface_models(limit: int = 50):
    """
    The main sync loop. Run this as a background job (cron, Celery, etc.).
    """
    print(f"Starting Hugging Face Sync Job (Targeting top {limit} TFLite models)...")
    api = HfApi()
    
    # Query HF for models with a library of 'tflite'
    hf_models = api.list_models(
            filter="tflite",
            limit=limit,
            full=True,  # Get full metadata
        )
    
    stats = {"models created": 0, "models updated": 0, "models skipped": 0,
                 "model versions created": 0, "model versions updated": 0, "model versions skipped": 0,}

    with Session(engine) as session:
        system_author_id = get_or_create_system_user(session)
        
        for hf_model in hf_models:
            repo_id = hf_model.id
            commit_sha = getattr(hf_model, "sha", None)
            
            if not commit_sha:
                print(f"[{repo_id}] Skipping: No commit SHA available.")
                stats["models skipped"] += 1
                continue
                
            print(f"[{repo_id}] Processing...")
            
            try:
                # 1. Parse Metadata
                # 'pipeline_tag' is HF's official task categorization (e.g., 'image-classification')
                task = getattr(hf_model, "pipeline_tag", "unknown")
                if task == "unknown" or task is None:
                    # If pipeline_tag is missing, try to interpolate from other metadata
                    task = interpolate_model_task(hf_model)
                license_type = extract_tag_value(getattr(hf_model, "tags", []), "license:")
                
                # 2. Create or Update the Parent Model Summary
                model = session.exec(select(MLModelDB).where(MLModelDB.hf_model_id == repo_id)).first()
                
                if not model:
                    model = MLModelDB(
                        name=repo_id.split("/")[-1].replace("-", " ").title(), # Format repo name nicely
                        slug=repo_id.replace("/", "-").lower(),
                        description=f"LiteRT model synced from Hugging Face from {repo_id}.",
                        category=ModelCategory.UTILITY, # Default categorization
                        author_id=system_author_id,
                        hf_model_id=repo_id,
                        task=task
                    )
                    session.add(model)
                    session.commit() # Commit early so we have the model.id for the version
                    session.refresh(model)
                    stats["models created"] += 1
                else:
                    # Optionally update dynamic fields like task if HF changed them
                    model.task = task
                    session.add(model)
                    session.commit()
                    stats["models updated"] += 1

                # 3. Check for Existing Version
                existing_version = session.exec(
                    select(ModelVersionDB)
                    .where(ModelVersionDB.model_id == model.id)
                    .where(ModelVersionDB.commit_sha == commit_sha)
                ).first()

                if existing_version:
                    print(f"[{repo_id}] Version {commit_sha[:7]} already exists. Skipping scanner.")
                    stats["model versions skipped"] += 1
                    continue

                # 4. Invoke the Heuristic Scanner (Strategies 1 & 2)
                version_payload = scan_hf_repo_for_version_assets(
                    repo_id=repo_id, 
                    commit_sha=commit_sha, 
                    license_type=license_type
                )

                if not version_payload:
                    print(f"[{repo_id}] No deployable TFLite assets found. Skipping version.")
                    stats["model versions skipped"] += 1
                    continue

                # 5. Commit the Version to PostgreSQL
                new_version = ModelVersionDB(
                    model_id=model.id,
                    version_name=version_payload["version_name"],
                    commit_sha=version_payload["commit_sha"],
                    is_hosted_by_us=version_payload["is_hosted_by_us"],
                    assets=version_payload["assets"], # The strict JSONB dictionary
                    pipeline_spec=version_payload.get("pipeline_spec"),
                    license_type=version_payload["license_type"],
                    is_commercial_safe=version_payload["is_commercial_safe"],
                    requires_commercial_warning=version_payload["requires_commercial_warning"],
                    file_size_bytes=version_payload["file_size_bytes"],
                    status=version_payload["status"]
                )

                session.add(new_version)
                session.commit()
                print(f"[{repo_id}] Success! Added version {new_version.version_name} (Status: {new_version.status})")
                stats["model versions created"] += 1
                
            except Exception as e:
                session.rollback() # Crucial: prevent one bad model from crashing the DB transaction state
                print(f"[{repo_id}] Error syncing model: {e}")
                stats["models skipped"] += 1
            
            # Rate limiting to respect Hugging Face API
            #time.sleep(0.5) 
            
    print("Hugging Face Sync Job complete.")
    return stats

def sync_single_model_version(repo_id: str, commit_sha: str):
    """
    Utility function to sync a single model version by repo_id and commit_sha.
    Useful for on-demand syncs or testing.
    """
    print(f"Starting sync of Hugging Face model {repo_id} at commit {commit_sha[:7]}...")
    api = HfApi()
    
    # Query HF for models with a library of 'tflite'
    hf_model = api.model_info(repo_id=repo_id, revision=commit_sha)

    with Session(engine) as session:
        system_author_id = get_or_create_system_user(session)
        
        try:
            # 1. Parse Metadata
            # 'pipeline_tag' is HF's official task categorization (e.g., 'image-classification')
            task = getattr(hf_model, "pipeline_tag", "unknown")
            if task == "unknown" or task is None:
                # If pipeline_tag is missing, try to interpolate from other metadata
                task = interpolate_model_task(hf_model)
            license_type = extract_tag_value(getattr(hf_model, "tags", []), "license:")
            
            # 2. Create or Update the Parent Model Summary
            model = session.exec(select(MLModelDB).where(MLModelDB.hf_model_id == repo_id)).first()
            
            if not model:
                model = MLModelDB(
                    name=repo_id.split("/")[-1].replace("-", " ").title(), # Format repo name nicely
                    slug=repo_id.replace("/", "-").lower(),
                    description=f"LiteRT model synced from Hugging Face from {repo_id}.",
                    category=ModelCategory.UTILITY, # Default categorization
                    author_id=system_author_id,
                    hf_model_id=repo_id,
                    task=task
                )
                session.add(model)
                session.commit() # Commit early so we have the model.id for the version
                session.refresh(model)
                logger.info(f"[{repo_id}] Created new model entry.")
            else:
                # Optionally update dynamic fields like task if HF changed them
                model.task = task
                session.add(model)
                session.commit()
                logger.info(f"[{repo_id}] Updated existing model entry.")

            # 3. Check for Existing Version
            existing_version = session.exec(
                select(ModelVersionDB)
                .where(ModelVersionDB.model_id == model.id)
                .where(ModelVersionDB.commit_sha == commit_sha)
            ).first()

            if existing_version:
                logger.info(f"[{repo_id}] Version {commit_sha[:7]} already exists. Skipping scanner.")
                return

            # 4. Invoke the Heuristic Scanner (Strategies 1 & 2)
            version_payload = scan_hf_repo_for_version_assets(
                repo_id=repo_id, 
                commit_sha=commit_sha, 
                license_type=license_type
            )

            if not version_payload:
                logger.info(f"[{repo_id}] No deployable TFLite assets found. Skipping version.")
                return

            # 5. Commit the Version to PostgreSQL
            new_version = ModelVersionDB(
                model_id=model.id,
                version_name=version_payload["version_name"],
                commit_sha=version_payload["commit_sha"],
                is_hosted_by_us=version_payload["is_hosted_by_us"],
                assets=version_payload["assets"], # The strict JSONB dictionary
                pipeline_spec=version_payload.get("pipeline_spec"),
                license_type=version_payload["license_type"],
                is_commercial_safe=version_payload["is_commercial_safe"],
                requires_commercial_warning=version_payload["requires_commercial_warning"],
                file_size_bytes=version_payload["file_size_bytes"],
                status=version_payload["status"]
            )

            session.add(new_version)
            session.commit()
            logger.info(f"[{repo_id}] Success! Added version {new_version.version_name} (Status: {new_version.status})")
            
        except Exception as e:
            session.rollback()
            logger.info(f"[{repo_id}] Error syncing model: {e}")
            return
        
        # Rate limiting to respect Hugging Face API
        #time.sleep(0.5) 
            
    print("Hugging Face Sync Job complete.")
    return

def run_sync(limit: int = settings.HF_SYNC_FETCH_LIMIT) -> Dict[str, int]:
    """
    Main entry point for the HuggingFace sync job.
    Fetches LiteRT models and syncs them to the database.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting HuggingFace LiteRT Model Sync")
        logger.info("=" * 60)
        
        # Sync to database
        stats = sync_huggingface_models(limit=limit)
        
        logger.info("=" * 60)
        logger.info("HuggingFace LiteRT Model Sync Completed Successfully")
        logger.info(f"Results: {stats}")
        logger.info("=" * 60)
        
        return stats
        
    except Exception as e:
        logger.error(f"Fatal error during sync: {e}")
        raise


if __name__ == "__main__":
    # Run the sync manually for cron job
    run_sync(limit = settings.HF_SYNC_FETCH_LIMIT)