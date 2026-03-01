import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel as PydanticBaseModel
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from pipeline_schema import PipelineConfig, PreprocessBlock, PostprocessBlock

# ==========================================
# 0. HELPERS
# ==========================================
def utc_now():
    return datetime.now(timezone.utc)

# ==========================================
# 1. ENUMS (Shared Constraints)
# ==========================================

class ModelCategory(str, Enum):
    UTILITY = "utility"             
    DIAGNOSTIC = "diagnostic"       
    PERFORMANCE = "performance"     
    FUN = "fun"                     
    OTHER = "other"

class DevicePlatform(str, Enum):
    ANDROID = "android"
    IOS = "ios"

# ==========================================
# 2. JSON COMPONENTS (The "Inner" Data)
# ==========================================
# Strict validation for our Dart/Flutter engine execution
# (PipelineConfig and related classes are imported from pipeline_schema.py)

# 1. Define the Strict Asset Contract
class AssetPointers(PydanticBaseModel):
    """
    Highly extensible pointer map. 
    Adding new asset types here requires ZERO Postgres migrations.
    """
    tflite: str  # The core execution binary is always required
    labels: Optional[str] = None
    tokenizer: Optional[str] = None
    vocab: Optional[str] = None
    anchors: Optional[str] = None # For complex object detection

# ==========================================
# 3. USER ENTITY
# ==========================================

class UserBase(SQLModel):
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    is_developer: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    hf_username: Optional[str] = Field(default=None, index=True)
    hf_verification_token: Optional[str] = Field(default=None)
    hf_access_token: Optional[str] = Field(default=None)

class UserDB(UserBase, table=True):
    __tablename__ = "users"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Relationships
    models: List["MLModelDB"] = Relationship(back_populates="author")

class UserRead(PydanticBaseModel):
    id: uuid.UUID
    username: str
    email: str
    is_developer: bool
    created_at: datetime
    hf_username: Optional[str] = None

# ==========================================
# 4. ML MODEL ENTITY (The "Product")
# ==========================================

class MLModelBase(SQLModel):
    name: str
    slug: Optional[str] = Field(index=True, unique=True)
    description: Optional[str] = None
    category: ModelCategory = Field(default=ModelCategory.UTILITY)

class MLModelDB(MLModelBase, table=True):
    __tablename__ = "ml_models"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    author_id: uuid.UUID = Field(foreign_key="users.id")
    hf_model_id: Optional[str] = Field(default=None, index=True)
    is_verified_official: bool = False
    
    tags: List[str] = Field(sa_column=Column(JSONB), default_factory=list)
    # Dynamic Task String (e.g., "image-classification") - Not an Enum!
    task: Optional[str] = Field(default=None, index=True)
    
    # Metrics
    total_download_count: int = 0  
    rating_weighted_avg: float = 0.0
    total_ratings: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    
    # Relationships
    author: UserDB = Relationship(back_populates="models")
    versions: List["ModelVersionDB"] = Relationship(back_populates="model")

class MLModelCreate(MLModelBase):
    description: str
    tags: List[str] = Field(default_factory=list)

class MLModelRead(PydanticBaseModel):
    id: uuid.UUID
    author_id: uuid.UUID
    name: str
    slug: Optional[str] = None
    description: str
    category: ModelCategory = ModelCategory.UTILITY
    tags: List[str]
    task: Optional[str] = None
    hf_model_id: Optional[str] = None
    is_verified_official: bool = False
    total_download_count: int
    total_ratings: int
    rating_weighted_avg: float
    created_at: datetime

# ==========================================
# 5. MODEL VERSION ENTITY (The "Logic")
# ==========================================

class ModelVersionBase(SQLModel):
    version_name: str = Field(index=True) # Usually the short commit_sha
    commit_sha: str = Field(index=True)   # The strict pointer lock
    
    # Strict Pointer Strategy enforcement
    is_hosted_by_us: bool = False
    assets: AssetPointers = Field(sa_column=Column(JSONB, nullable=False))
    
    # Commercial Safety Check
    license_type: str = Field(default="unknown")
    is_commercial_safe: bool = False
    requires_commercial_warning: bool = False
    file_size_bytes: int = 0
    
    # State tracking ("unconfigured" vs "configured")
    status: str = Field(default="unconfigured", index=True)
    changelog: Optional[str] = None

class ModelVersionDB(ModelVersionBase, table=True):
    __tablename__ = "model_versions"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    model_id: uuid.UUID = Field(foreign_key="ml_models.id")
    
    pipeline_spec: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    is_supported: bool = Field(default=False)
    unsupported_reason: Optional[str] = None

    published_at: datetime = Field(default_factory=utc_now)
    download_count: int = 0
    num_ratings: int = 0
    rating_avg: float = 0
        
    # Relationships
    model: MLModelDB = Relationship(back_populates="versions")
    logs: List["InferenceLogDB"] = Relationship(back_populates="version")

class ModelVersionCreate(ModelVersionBase):
    # Validated strictly through Pydantic upon creation/update
    pipeline_spec: Optional[PipelineConfig] = None

class ModelVersionRead(PydanticBaseModel):
    id: uuid.UUID
    model_id: uuid.UUID
    version_name: str
    commit_sha: str
    is_hosted_by_us: bool = False
    assets: AssetPointers
    license_type: str = "unknown"
    is_commercial_safe: bool = False
    requires_commercial_warning: bool = False
    file_size_bytes: int = 0
    status: str = "unconfigured"
    changelog: Optional[str] = None
    pipeline_spec: Optional[PipelineConfig] = None
    published_at: datetime
    download_count: int
    num_ratings: int
    rating_avg: float
    is_supported: bool
    unsupported_reason: Optional[str] = None

class ModelVersionUpdate(PydanticBaseModel):
    # This invokes our strict PipelineConfig rules
    pipeline_spec: Optional[PipelineConfig] = None
    status: str

# ==========================================
# 6. TELEMETRY
# ==========================================

class InferenceLogDB(SQLModel, table=True):
    __tablename__ = "inference_logs"
    id: Optional[int] = Field(default=None, primary_key=True) 
    model_version_id: uuid.UUID = Field(foreign_key="model_versions.id")
    timestamp: datetime = Field(default_factory=utc_now, index=True)
    
    device_model: str
    platform: DevicePlatform
    total_inference_ms: int
    success: bool
    
    version: ModelVersionDB = Relationship(back_populates="logs")

class InferenceLogCreate(SQLModel):
    device_model: str
    platform: DevicePlatform
    total_inference_ms: int
    success: bool