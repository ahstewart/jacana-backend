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
    framework: Literal["tflite", "pytorch_lite", "onnx", "mediapipe_litert"] = Field(default="tflite")
    source_repository: Optional[str] = None

class TensorDefinition(BaseModel):
    name: str = Field(description="The exact name of the tensor.")
    shape: List[int] = Field(description="The tensor shape, e.g., [1, 224, 224, 3].")
    dtype: Literal["float32", "uint8", "int8", "int32"] = Field(description="The data type of the tensor. Use int32 for token ID tensors in text generation models.")

# ==========================================
# 2. PREPROCESSING PARAMS
# ==========================================
class ResizeImageParams(BaseModel):
    height: int
    width: int
    method: Literal["bilinear", "nearest_neighbor"] = Field(default="bilinear")

class NormalizeParams(BaseModel):
    method: Literal["mean_stddev", "scale_shift", "scale_div"] = Field(default="mean_stddev")
    mean: Optional[List[float]] = None
    stddev: Optional[List[float]] = None
    scale: Optional[float] = None
    shift: Optional[float] = None
    value: Optional[float] = None   # used by scale_div (e.g. 255.0)
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

class DecodeSegmentationMaskParams(BaseModel):
    num_classes: int
    argmax_axis: int = -1          # axis that holds per-class scores (usually last)
    color_map: Optional[str] = None  # "pascal_voc" | "cityscapes" | "ade20k" | None (auto)

# ==========================================
# ASR PARAMS
# ==========================================
class ResampleAudioParams(BaseModel):
    target_sample_rate: int = 16000   # Hz — must match the model's baked-in rate
    max_duration_s: float = 10.0      # trim/zero-pad to this length
    normalize: bool = True            # peak-normalize PCM to [-1.0, 1.0]

class CtcDecodeParams(BaseModel):
    blank_id: int = 0                 # CTC blank token ID (wav2vec2: 0)
    word_delimiter: str = "|"         # token used as word boundary (wav2vec2: "|")
    vocabulary_url: Optional[str] = None  # URL/path to vocab file if not in assets

# ==========================================
# TEXT GENERATION PARAMS
# ==========================================
class TokenizeParams(BaseModel):
    max_length: int = Field(default=512, description="Maximum sequence length after tokenization.")
    padding: bool = Field(default=True, description="Pad sequences shorter than max_length.")
    truncation: bool = Field(default=True, description="Truncate sequences longer than max_length.")
    add_special_tokens: bool = Field(default=True, description="Add model-specific special tokens (e.g. [CLS]/[SEP] for BERT, <s>/</s> for RoBERTa).")
    vocab_file: Optional[str] = Field(default=None, description="Filename for vocab.txt (WordPiece/BERT-style). Auto-detected if omitted.")
    tokenizer_file: Optional[str] = Field(default=None, description="Filename for tokenizer.json (HuggingFace BPE format). Auto-detected if omitted.")

class GenerateParams(BaseModel):
    mode: Literal["single_pass", "autoregressive"] = Field(
        default="single_pass",
        description="single_pass: model runs once and outputs the full sequence (seq2seq). autoregressive: decode loop, model runs repeatedly appending one token at a time (causal LM)."
    )
    max_new_tokens: int = Field(default=128, description="Maximum number of tokens to generate (autoregressive only).")
    temperature: float = Field(default=1.0, description="Sampling temperature. 1.0 = neutral, <1.0 = sharper, >1.0 = more random.")
    do_sample: bool = Field(default=False, description="If False, use greedy decoding (argmax). If True, use temperature sampling.")
    eos_token_id: Optional[int] = Field(default=None, description="Token ID that signals end of generation. Generation stops when this token is produced.")

class DecodeTokensParams(BaseModel):
    skip_special_tokens: bool = Field(default=True, description="Remove special tokens (e.g. [CLS], [SEP], <pad>) from the decoded output string.")

class MediaPipeGenerateParams(BaseModel):
    model_type: Literal["gemmaIt", "general", "deepSeek", "qwen", "llama", "hammer"] = Field(
        default="gemmaIt",
        description="Maps to flutter_gemma ModelType. Use 'qwen' for Qwen models, 'llama' for Llama, 'deepSeek' for DeepSeek, 'gemmaIt' for Gemma instruction-tuned, 'general' for others."
    )
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    random_seed: Optional[int] = None

# ==========================================
# 4. THE STEP WRAPPERS
# ==========================================
class PreprocessStep(BaseModel):
    step: Literal["resize_image", "normalize", "format", "tokenize", "resample_audio"]
    params: Union[ResizeImageParams, NormalizeParams, FormatParams, TokenizeParams, ResampleAudioParams]

class PostprocessStep(BaseModel):
    step: Literal["apply_activation", "map_labels", "filter_by_score", "decode_boxes", "apply_nms", "generate", "decode_tokens", "mediapipe_generate", "decode_segmentation_mask", "ctc_decode"]
    params: Union[ApplyActivationParams, MapLabelsParams, FilterByScoreParams, DecodeBoxesParams, ApplyNMSParams, GenerateParams, DecodeTokensParams, MediaPipeGenerateParams, DecodeSegmentationMaskParams, CtcDecodeParams]

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