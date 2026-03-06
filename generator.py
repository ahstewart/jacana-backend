import os
import time
import tempfile
import logging
import httpx
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from pydantic import ValidationError, BaseModel, Field
from typing import Optional
from sqlmodel import Session, select
from huggingface_hub import HfApi
from google import genai
from google.genai import types
from tflite_support import metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Database and Schema import
from database import engine
from schema import MLModelDB, ModelVersionDB
from pipeline_schema import PipelineConfig
from config import get_settings
from validator import validate_and_correct_pipeline


# Load environment variables (ensure OPENAI_API_KEY is in your .env file)
settings = get_settings()

# Initialize the Gemini client for the LLM extraction
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# THE ENVELOPE PATTERN
class PipelineGenerationResult(BaseModel):
    is_supported: bool = Field(description="Set to true ONLY if the model's task and required operations can be perfectly mapped to the provided pipeline schema. Set to false if the model requires unsupported operations (e.g., audio processing, NLP, custom C++ ops) or lacks sufficient metadata.")
    reasoning: str = Field(description="A brief explanation of why the model is supported or unsupported.")
    config: Optional[PipelineConfig] = Field(default=None, description="The generated pipeline configuration. Must be provided if is_supported is true.")

def fetch_hf_readme(repo_id: str, commit_sha: str) -> str:
    """
    Fetches the README.md (Model Card) directly from the Hugging Face repo.
    """
    api = HfApi()
    url = f"https://huggingface.co/{repo_id}/resolve/{commit_sha}/README.md"
    logger.debug("Fetching README: %s", url)
    try:
        response = httpx.get(url, timeout=settings.HF_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
        if response.status_code == 200:
            logger.debug("README fetched (%d chars)", len(response.text))
            return response.text
        logger.warning("README not found for %s (HTTP %d)", repo_id, response.status_code)
        return "No README found."
    except Exception as e:
        logger.warning("Failed to fetch README for %s: %s", repo_id, e)
        return "Failed to fetch README."

def fetch_hf_model_card(repo_id: str) -> str:
    """
    Fetches the structured Model Card metadata (YAML frontmatter and tags)
    from the Hugging Face Hub API.
    """
    api = HfApi()
    logger.debug("Fetching model card for %s...", repo_id)
    try:
        info = api.model_info(repo_id=repo_id)
        card_data = getattr(info, 'cardData', {})
        result = str(card_data) if card_data else "No structured Model Card data found."
        logger.debug("Model card fetched (%d chars)", len(result))
        return result
    except Exception as e:
        logger.warning("Failed to fetch model card for %s: %s", repo_id, e)
        return "Failed to fetch Model Card data."

def fetch_tflite_metadata(tflite_url: str) -> str:
    """
    Attempts to download the TFLite file and extract embedded FlatBuffer metadata.
    """
    logger.debug("Downloading TFLite for metadata extraction: %s", tflite_url)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as tmp_file:
            t0 = time.monotonic()
            download_deadline = t0 + settings.TFLITE_DOWNLOAD_TIMEOUT_SECONDS
            downloaded = 0
            last_progress_log = t0
            with httpx.stream("GET", tflite_url, follow_redirects=True,
                              timeout=settings.HF_FETCH_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=8192):
                    if time.monotonic() > download_deadline:
                        raise TimeoutError(
                            f"Metadata download timed out after {settings.TFLITE_DOWNLOAD_TIMEOUT_SECONDS}s "
                            f"({downloaded / 1024 / 1024:.1f} MB received)"
                        )
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_progress_log >= 5.0:
                        logger.debug("Downloading for metadata... %.1f MB received", downloaded / 1024 / 1024)
                        last_progress_log = now
            tmp_path = tmp_file.name
        logger.debug("TFLite downloaded (%.1f MB) in %.1fs",
                     os.path.getsize(tmp_path) / 1024 / 1024, time.monotonic() - t0)

        displayer = metadata.MetadataDisplayer.with_model_file(tmp_path)
        metadata_text = displayer.get_metadata_json()
        os.remove(tmp_path)

        result = metadata_text if metadata_text else "No embedded metadata found in TFLite file."
        logger.debug("Metadata extracted (%d chars)", len(result))
        return result

    except Exception as e:
        logger.warning("Failed to extract TFLite metadata: %s", e)
        return "Metadata extraction failed."

def generate_pipeline_config(
    task: str,
    readme_text: str,
    metadata_text: str,
    model_card_text: str,
    validation_error: str | None = None,
) -> PipelineGenerationResult | None:
    """
    Feeds the unstructured text to Gemini and forces it to return 
    a strict JSON object matching our PipelineGenerationResult wrapper.
    """
    system_prompt = f"""
    You are an expert Edge AI engineer responsible for writing metadata files for AI models. These files will be used to run inference on these models on edge devices.
    Your job is to analyze the following Hugging Face Model Card, README, and TFLite Metadata for the AI model, and generate a strict configuration file for mobile deployment.

    Model Task: {task}

    IMPORTANT:
    Make your best attempt at generating a pipeline config. Do not reject a model just because you are
    uncertain about exact tensor shapes or dtypes — those will be verified and auto-corrected against
    the actual TFLite binary after you generate the config. Focus on getting the structure right:
    correct step types, correct interpretation values, correct routing between tensors.

    Only set 'is_supported' to false for models that are fundamentally incompatible with the schema —
    i.e. ones whose task cannot be expressed using the available step types at all:
    - Audio/speech models (no audio preprocessing steps exist in the schema)
    - Multi-modal models requiring simultaneous image + text inputs
    - Models that require custom C++ ops with no standard TFLite equivalent

    Instructions if supported:
    1. TENSORS: explicitly define 'inputs' and 'outputs' using exact tensor names, shapes, and dtypes.
       - Use dtype "int32" for token ID tensors (text generation models).
       - Use dtype "float32" for logit/score tensors.
    2. ROUTING: 'input_name' and 'source_tensors' must match the defined tensors.
    3. PREPROCESSING: Create a LIST of PreprocessBlock objects. Each block has:
       - input_name: str (name of the input tensor)
       - expects_type: Literal["image", "audio", "text"]
       - steps: List of PreprocessStep objects

       For TEXT GENERATION models, use expects_type "text" and a single "tokenize" step:
         step: "tokenize"
         params: {{max_length: <N>, padding: true, truncation: true, add_special_tokens: true}}
         where max_length MUST equal the sequence dimension from the TFLite input tensor shape (e.g. if input shape is [1, 64], use max_length: 64).

    4. POSTPROCESSING: Create a LIST of PostprocessBlock objects. Each block has:
       - output_name: str (logical name for the output)
       - interpretation: str — use "text_generation" for all text generation models
       - source_tensors: List of output tensor names
       - coordinate_format: Optional[str] (for detection models only)
       - steps: List of PostprocessStep objects

       For SINGLE-PASS seq2seq models (e.g., translation, summarization — model outputs full token ID sequence in one call):
         steps:
           - step: "decode_tokens", params: {{skip_special_tokens: true}}

       For AUTOREGRESSIVE causal LM models (e.g., GPT-style — model outputs logits and must be called in a decode loop):
         steps:
           - step: "generate", params: {{mode: "autoregressive", max_new_tokens: 128, temperature: 1.0, do_sample: false, eos_token_id: <EOS_ID>}}
           - step: "decode_tokens", params: {{skip_special_tokens: true}}

    5. You MUST map requirements ONLY to the exact structures and allowed literals defined in the schema.

    EXAMPLE — single-pass seq2seq (T5-style translation, input/output are token IDs):
    {{
      "metadata": [{{"model_task": "text_generation", ...}}],
      "inputs": [{{"name": "input_ids", "shape": [1, 128], "dtype": "int32"}}],
      "outputs": [{{"name": "output_ids", "shape": [1, 128], "dtype": "int32"}}],
      "preprocessing": [{{"input_name": "input_ids", "expects_type": "text", "steps": [{{"step": "tokenize", "params": {{"max_length": 128, "padding": true, "truncation": true, "add_special_tokens": true}}}}]}}],
      "postprocessing": [{{"output_name": "generated_text", "interpretation": "text_generation", "source_tensors": ["output_ids"], "steps": [{{"step": "decode_tokens", "params": {{"skip_special_tokens": true}}}}]}}]
    }}

    EXAMPLE — autoregressive causal LM (GPT-2 style, outputs logits over vocab):
    {{
      "metadata": [{{"model_task": "text_generation", ...}}],
      "inputs": [{{"name": "input_ids", "shape": [1, 512], "dtype": "int32"}}],
      "outputs": [{{"name": "logits", "shape": [1, 512, 50257], "dtype": "float32"}}],
      "preprocessing": [{{"input_name": "input_ids", "expects_type": "text", "steps": [{{"step": "tokenize", "params": {{"max_length": 512, "padding": false, "truncation": true, "add_special_tokens": true}}}}]}}],
      "postprocessing": [{{"output_name": "generated_text", "interpretation": "text_generation", "source_tensors": ["logits"], "steps": [{{"step": "generate", "params": {{"mode": "autoregressive", "max_new_tokens": 128, "temperature": 1.0, "do_sample": false, "eos_token_id": 50256}}}}, {{"step": "decode_tokens", "params": {{"skip_special_tokens": true}}}}]}}]
    }}

    EXAMPLE — MediaPipe LLM Inference (models with .litertlm or .task asset, e.g. Qwen2.5, Gemma):
    Use framework "mediapipe_litert". Tensor I/O is abstracted by the API — use a placeholder input/output tensor.
    {{
      "metadata": [{{"schema_version": "1.0.0", "model_name": "...", "model_version": "...", "model_task": "text_generation", "framework": "mediapipe_litert", "source_repository": "..."}}],
      "inputs": [{{"name": "input_text", "shape": [1], "dtype": "int32"}}],
      "outputs": [{{"name": "output_text", "shape": [1], "dtype": "int32"}}],
      "preprocessing": [{{"input_name": "input_text", "expects_type": "text", "steps": [{{"step": "tokenize", "params": {{"max_length": 512, "padding": false, "truncation": true, "add_special_tokens": true}}}}]}}],
      "postprocessing": [{{"output_name": "generated_text", "interpretation": "text_generation", "source_tensors": ["output_text"], "steps": [{{"step": "mediapipe_generate", "params": {{"model_type": "<gemmaIt|qwen|llama|deepSeek|general>", "max_tokens": 512, "temperature": 0.8, "top_k": 40, "top_p": 0.9}}}}]}}]
    }}

    RULE: Do NOT set is_supported to false for models whose asset file is .litertlm or .task.
    These models MUST use framework "mediapipe_litert" and the "mediapipe_generate" postprocessing step.

    SEGMENTATION RULES:
    There are two segmentation output formats — choose based on the output tensor shape and dtype:

    FORMAT A — Float32 per-class scores [1, H, W, C] where C > 1:
      Use interpretation "segmentation_mask" with a "decode_segmentation_mask" step.
      Set num_classes to C. Preprocessing: resize → normalize (mean_stddev or scale_shift) → format (target_dtype: "float32").
      Use color_map "pascal_voc" for 21-class VOC, "cityscapes" for 19-class Cityscapes, null otherwise.

    FORMAT B — Uint8/int32 class indices [1, H, W] or [1, H, W, 1] (already argmaxed):
      Also use interpretation "segmentation_mask" with a "decode_segmentation_mask" step — the inference
      engine auto-detects whether argmax is needed from the tensor layout.
      Set num_classes to the number of semantic classes the model was trained on (from the model card).
      CRITICAL: For uint8 input models, OMIT the normalize step entirely — quantized models have
      normalization baked into their weights. Use target_dtype: "uint8" in the format step.

    Always set model_task to "semantic_segmentation".
    Always read input shape, output shape, and dtype from the TFLite metadata — do NOT guess or use
    values from the model card source code. The TFLite metadata is authoritative.

    EXAMPLE A — DeepLab v3 MobileNet (float32, Pascal VOC, 21 classes):
    {{
      "metadata": [{{"schema_version": "1.0.0", "model_name": "DeepLab v3", "model_version": "1.0",
        "model_task": "semantic_segmentation", "framework": "tflite"}}],
      "inputs": [{{"name": "sub_2", "shape": [1, 257, 257, 3], "dtype": "float32"}}],
      "outputs": [{{"name": "ResizeBilinear_2", "shape": [1, 257, 257, 21], "dtype": "float32"}}],
      "preprocessing": [{{
        "input_name": "sub_2", "expects_type": "image",
        "steps": [
          {{"step": "resize_image", "params": {{"height": 257, "width": 257, "method": "bilinear"}}}},
          {{"step": "normalize", "params": {{"method": "scale_shift", "scale": 0.007843137, "shift": -1.0}}}},
          {{"step": "format", "params": {{"target_dtype": "float32", "color_space": "RGB", "data_layout": "NHWC"}}}}
        ]
      }}],
      "postprocessing": [{{
        "output_name": "segmentation_mask", "interpretation": "segmentation_mask",
        "source_tensors": ["ResizeBilinear_2"],
        "steps": [
          {{"step": "decode_segmentation_mask", "params": {{"num_classes": 21, "argmax_axis": -1, "color_map": "pascal_voc"}}}}
        ]
      }}]
    }}

    EXAMPLE B — Quantized segmentation model (uint8 input, class-index output [1, H, W], 21 classes):
    {{
      "metadata": [{{"schema_version": "1.0.0", "model_name": "DeepLab v3 Quantized", "model_version": "1.0",
        "model_task": "semantic_segmentation", "framework": "tflite"}}],
      "inputs": [{{"name": "input", "shape": [1, 257, 257, 3], "dtype": "uint8"}}],
      "outputs": [{{"name": "output", "shape": [1, 257, 257], "dtype": "uint8"}}],
      "preprocessing": [{{
        "input_name": "input", "expects_type": "image",
        "steps": [
          {{"step": "resize_image", "params": {{"height": 257, "width": 257, "method": "bilinear"}}}},
          {{"step": "format", "params": {{"target_dtype": "uint8", "color_space": "RGB", "data_layout": "NHWC"}}}}
        ]
      }}],
      "postprocessing": [{{
        "output_name": "segmentation_mask", "interpretation": "segmentation_mask",
        "source_tensors": ["output"],
        "steps": [
          {{"step": "decode_segmentation_mask", "params": {{"num_classes": 21, "argmax_axis": -1, "color_map": "pascal_voc"}}}}
        ]
      }}]
    }}
    """

    user_prompt = f"""
    Analyze this model and generate a complete, valid PipelineConfig with ALL required fields:
    - metadata: A list with one MetadataBlock containing model info
    - inputs: A list of TensorDefinition objects for all input tensors
    - outputs: A list of TensorDefinition objects for all output tensors
    - preprocessing: A LIST of PreprocessBlock objects (each block contains input_name, expects_type, and steps)
    - postprocessing: A LIST of PostprocessBlock objects (each block contains output_name, interpretation, source_tensors, and steps)

    --- TFLITE METADATA ---
    {metadata_text}

    --- MODEL CARD METADATA ---
    {model_card_text}

    --- README ---
    {readme_text}
    """

    if validation_error:
        user_prompt += f"""
--- PREVIOUS ATTEMPT VALIDATION ERROR ---
The previously generated pipeline config failed TFLite inference validation with this error:

{validation_error}

Fix the pipeline config to resolve this error. Pay close attention to input shape,
input dtype (float32 vs uint8), output shape, and preprocessing steps.
"""

    logger.info("Calling LLM (model=%s, with_error_feedback=%s, timeout=%ds)...",
                settings.PIPELINE_GENERATION_MODEL, bool(validation_error), settings.LLM_TIMEOUT_SECONDS)
    t0 = time.monotonic()

    def _call_llm():
        return client.models.generate_content(
            model=settings.PIPELINE_GENERATION_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_json_schema=PipelineGenerationResult.model_json_schema(),
            ),
            contents=user_prompt,
        )

    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_call_llm)
        try:
            response = future.result(timeout=settings.LLM_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            logger.error("LLM call timed out after %ds", settings.LLM_TIMEOUT_SECONDS)
            return None
        finally:
            executor.shutdown(wait=False)

        elapsed = time.monotonic() - t0
        result = PipelineGenerationResult.model_validate_json(response.text)
        logger.info("LLM responded in %.1fs — is_supported=%s", elapsed, result.is_supported)
        if result.reasoning:
            logger.debug("LLM reasoning: %s", result.reasoning[:300])
        return result

    except Exception as e:
        logger.error("LLM generation failed: %s", e, exc_info=True)
        return None
    

def run_generator_for_version(version: ModelVersionDB, model: MLModelDB, session: Session) -> bool:
    """
    Executes the pipeline generation for a single version and commits to DB.
    Returns True if generation was successfully processed (even if unsupported), False if it failed.
    """
    label = f"{model.name} / {version.version_name}"
    logger.info("[%s] ── Starting pipeline generation (task=%s) ──", label, model.task)

    # 1. Gather Context
    logger.debug("[%s] Fetching README (repo=%s, sha=%s)...", label, model.hf_model_id, version.commit_sha)
    readme_text = fetch_hf_readme(model.hf_model_id, version.commit_sha)

    logger.debug("[%s] Fetching model card...", label)
    model_card_text = fetch_hf_model_card(model.hf_model_id)

    tflite_url = version.assets.get("tflite")
    if tflite_url:
        logger.debug("[%s] Fetching TFLite metadata...", label)
    else:
        logger.warning("[%s] No TFLite URL in assets — skipping metadata extraction", label)
    metadata_text = fetch_tflite_metadata(tflite_url) if tflite_url else "No TFLite URL found."

    # 2. Ask the LLM to generate the configuration
    logger.info("[%s] Sending context to LLM...", label)
    result = generate_pipeline_config(model.task, readme_text, metadata_text, model_card_text)

    # 3. Handle the Result
    if result:
        if result.is_supported and result.config:
            logger.info("[%s] LLM produced a config (%d pre-steps, %d post-steps)",
                        label, len(result.config.preprocessing), len(result.config.postprocessing))

            # Validate and auto-correct the pipeline via actual TFLite inference
            validation_mode = settings.PIPELINE_VALIDATION_MODE
            if validation_mode == "none" or not tflite_url:
                if validation_mode == "none":
                    logger.info("[%s] Validation mode=none — skipping", label)
                else:
                    logger.warning("[%s] No TFLite URL — cannot validate, skipping", label)
                new_status = "unverified"
                status_reason = None
            else:
                # Shared retry loop for both "strict" and "loose"
                validation_passed = False
                last_error = None
                last_retryable = False  # True = structural failure; False = environment failure
                last_corrected = result.config
                for attempt in range(settings.MAX_VALIDATION_RETRIES):
                    logger.info("[%s] Validation attempt %d/%d (mode=%s)...",
                                label, attempt + 1, settings.MAX_VALIDATION_RETRIES, validation_mode)
                    ok, error, retryable, corrected = validate_and_correct_pipeline(result.config, tflite_url)
                    last_corrected = corrected
                    if ok:
                        result.config = corrected
                        validation_passed = True
                        logger.info("[%s] Validation passed", label)
                        break
                    last_error = error
                    last_retryable = retryable
                    logger.warning("[%s] Validation attempt %d failed: %s", label, attempt + 1, error)
                    if not retryable:
                        logger.warning("[%s] Error is not retryable — skipping further attempts", label)
                        break
                    if attempt < settings.MAX_VALIDATION_RETRIES - 1:
                        logger.info("[%s] Re-asking LLM with validation error context...", label)
                        retry_result = generate_pipeline_config(
                            model.task, readme_text, metadata_text, model_card_text,
                            validation_error=error,
                        )
                        if not (retry_result and retry_result.is_supported and retry_result.config):
                            last_error = "LLM failed to produce a valid config on retry."
                            logger.error("[%s] LLM retry produced no usable config", label)
                            break
                        result = retry_result

                if validation_passed:
                    new_status = "supported"
                    status_reason = None
                else:
                    # Structural failures (wrong op, bad shape) → broken; environment failures
                    # (timeout, model too large) → unverified (pipeline may still work on device)
                    new_status = "broken" if last_retryable else "unverified"
                    status_reason = f"TFLite validation failed: {last_error}"
                    result.config = last_corrected
                    logger.warning("[%s] Validation failed — storing as %s. Reason: %s",
                                   label, new_status, last_error)

            logger.info("[%s] Saving pipeline config to DB (status=%s)...", label, new_status)
            version.pipeline_spec = result.config.model_dump(mode='json')
            version.status = new_status
            version.unsupported_reason = status_reason
        else:
            logger.info("[%s] LLM rejected model — reason: %s", label, result.reasoning)
            version.status = "unsupported"
            version.unsupported_reason = result.reasoning

        session.add(version)
        session.commit()
        logger.info("[%s] ── Done (status=%s) ──", label, version.status)
        return True
    else:
        logger.error("[%s] LLM returned no result — leaving as unconfigured", label)
        return False



def run_generator_for_huggingface_model(repo_id: str, commit_sha: str):
    """
    A helper function to run the entire generator process for a specific Hugging Face model version.
    This can be used for manual triggering or testing with specific models.
    """
    with Session(engine) as session:
        model = session.exec(select(MLModelDB).where(MLModelDB.hf_model_id == repo_id)).first()
        if not model:
            print(f"Model with repo_id {repo_id} not found in database.")
            return
        
        version = session.exec(
            select(ModelVersionDB)
            .where(ModelVersionDB.model_id == model.id)
            .where(ModelVersionDB.commit_sha == commit_sha)
        ).first()

        if not version:
            print(f"Version with commit_sha {commit_sha} for model {repo_id} not found in database.")
            return
        
        run_generator_for_version(version, model, session)


def _process_version_isolated(version_id: int, model_id: int) -> bool:
    """
    Worker function safe to call from a thread pool.
    Opens its own DB session so threads don't share state.
    """
    with Session(engine) as session:
        version = session.get(ModelVersionDB, version_id)
        model = session.get(MLModelDB, model_id)
        if not version or not model:
            print(f"  -> Could not load version {version_id} or model {model_id} from DB.")
            return False
        print(f"\nProcessing: {model.name} (Version: {version.version_name})")
        return run_generator_for_version(version, model, session)


def process_all_unconfigured():
    """
    The main job loop. Finds all unconfigured versions, gathers context, 
    asks the LLM for a config, and saves it if successful.
    """
    print("Starting LLM Pipeline Generator...")

    with Session(engine) as session:
        statement = select(ModelVersionDB, MLModelDB).join(MLModelDB).where(ModelVersionDB.status == "unconfigured")
        results = session.exec(statement).all()
        pairs = [(v.id, m.id) for v, m in results]

    if not pairs:
        print("No unconfigured models found. It's a miracle!")
        return

    print(f"Processing {len(pairs)} model(s) with up to {settings.MAX_GENERATOR_WORKERS} parallel workers...")
    with ThreadPoolExecutor(max_workers=settings.MAX_GENERATOR_WORKERS) as executor:
        futures = {executor.submit(_process_version_isolated, vid, mid): (vid, mid) for vid, mid in pairs}
        for future in as_completed(futures):
            vid, mid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  -> Unhandled error for version {vid}: {e}")


_SEGMENTATION_TASKS = {"image-segmentation", "semantic-segmentation", "segmentation", "image_segmentation", "semantic_segmentation"}

def retry_unsupported_segmentation():
    """
    Finds all model versions that were previously rejected and belong to a
    segmentation-task model, then re-runs pipeline generation for each one.
    Call this after updating the generator rules to allow segmentation.
    """
    print("Retrying previously-rejected segmentation models...")

    with Session(engine) as session:
        statement = (
            select(ModelVersionDB, MLModelDB)
            .join(MLModelDB)
            .where(ModelVersionDB.status == "unsupported")
            .where(MLModelDB.task.in_(list(_SEGMENTATION_TASKS)))
        )
        results = session.exec(statement).all()

        pairs = [(v.id, m.id) for v, m in results]

    if not pairs:
        print("No unsupported segmentation models found.")
        return

    print(f"Retrying {len(pairs)} segmentation model(s) with up to {settings.MAX_GENERATOR_WORKERS} parallel workers...")
    with ThreadPoolExecutor(max_workers=settings.MAX_GENERATOR_WORKERS) as executor:
        futures = {executor.submit(_process_version_isolated, vid, mid): (vid, mid) for vid, mid in pairs}
        for future in as_completed(futures):
            vid, mid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  -> Unhandled error for version {vid}: {e}")


if __name__ == "__main__":
    #process_all_unconfigured()
    run_generator_for_huggingface_model(repo_id="openai-community/gpt2", commit_sha="607a30d783dfa663caf39e06633721c8d4cfcd7e")