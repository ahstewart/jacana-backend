from pydantic import BaseModel, Field
from typing import List, Literal, Union, Optional

# ==========================================
# 1. METADATA & TENSOR DEFINITIONS
# ==========================================
class MetadataBlock(BaseModel):
    schema_version: str = Field(default="1.0.0")
    model_name: str
    model_version: str
    model_task: str
    framework: Literal["tflite", "pytorch_lite", "onnx"] = Field(default="tflite")
    source_repository: Optional[str] = None

class TensorDefinition(BaseModel):
    name: str = Field(description="The exact name of the tensor.")
    shape: List[int] = Field(description="The tensor shape, e.g., [1, 224, 224, 3].")
    dtype: Literal["float32", "uint8", "int8"] = Field(description="The data type of the tensor.")

# ==========================================
# 2. PREPROCESSING PARAMS
# ==========================================
class ResizeImageParams(BaseModel):
    height: int
    width: int
    method: Literal["bilinear", "nearest_neighbor"] = Field(default="bilinear")

class NormalizeParams(BaseModel):
    method: Literal["mean_stddev", "scale_shift"] = Field(default="mean_stddev")
    mean: Optional[List[float]] = None
    stddev: Optional[List[float]] = None
    scale: Optional[float] = None
    shift: Optional[float] = None
    color_space: Optional[str] = Field(default="RGB")

class FormatParams(BaseModel):
    target_dtype: Literal["float32", "uint8", "int8"]
    color_space: str = Field(default="RGB")
    data_layout: Literal["NHWC", "NCHW"] = Field(default="NHWC")

# ==========================================
# 3. POSTPROCESSING PARAMS
# ==========================================
class ApplyActivationParams(BaseModel):
    function: Literal["softmax", "sigmoid", "relu"]

class MapLabelsParams(BaseModel):
    labels_url: str = Field(description="URL or relative path to the labels file.")
    class_tensor: str = Field(description="The name of the tensor holding class indices or scores.")
    top_k: int = Field(default=5)
    label_offset: Optional[int] = Field(default=0)

class FilterByScoreParams(BaseModel):
    threshold: float
    score_tensor: str
    num_detections_tensor: Optional[str] = None

class DecodeBoxesParams(BaseModel):
    box_tensor: str

class ApplyNMSParams(BaseModel):
    iou_threshold: float
    score_threshold: float
    box_tensor: str
    score_tensor: str

# ==========================================
# 4. THE STEP WRAPPERS
# ==========================================
class PreprocessStep(BaseModel):
    step: Literal["resize_image", "normalize", "format"]
    params: Union[ResizeImageParams, NormalizeParams, FormatParams]

class PostprocessStep(BaseModel):
    step: Literal["apply_activation", "map_labels", "filter_by_score", "decode_boxes", "apply_nms"]
    params: Union[ApplyActivationParams, MapLabelsParams, FilterByScoreParams, DecodeBoxesParams, ApplyNMSParams]

# ==========================================
# 5. THE MASTER CONFIGURATION BLOCKS
# ==========================================
class PreprocessBlock(BaseModel):
    input_name: str = Field(description="Links to the input tensor defined in inputs.")
    expects_type: Literal["image", "audio", "text"] = Field(default="image")
    steps: List[PreprocessStep]

class PostprocessBlock(BaseModel):
    output_name: str = Field(description="Logical name for the final output.")
    interpretation: str = Field(description="e.g., 'classification_scores' or 'detection_boxes_scores_classes'")
    source_tensors: List[str] = Field(description="Links to the raw output tensors.")
    coordinate_format: Optional[str] = Field(default=None, description="e.g., 'normalized_ymin_xmin_ymax_xmax'")
    steps: List[PostprocessStep]

class PipelineConfig(BaseModel):
    """The master contract representing the YAML file."""
    metadata: List[MetadataBlock]
    inputs: List[TensorDefinition]
    outputs: List[TensorDefinition]
    preprocessing: List[PreprocessBlock]
    postprocessing: List[PostprocessBlock]