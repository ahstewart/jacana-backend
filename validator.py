import os
import time
import tempfile
import logging
import concurrent.futures

import numpy as np
import httpx
from ai_edge_litert.interpreter import Interpreter

from pipeline_schema import PipelineConfig
from config import get_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _np_dtype_to_str(np_dtype) -> str:
    mapping = {np.float32: "float32", np.uint8: "uint8", np.int8: "int8", np.int32: "int32"}
    return mapping.get(np_dtype, "float32")


def _make_dummy_input(shape, dtype) -> np.ndarray:
    """Returns a safe non-zero dummy tensor for the given shape and dtype."""
    if dtype == np.uint8:
        return np.full(shape, 128, dtype=dtype)   # mid-range, avoids saturation
    if dtype == np.int32:
        return np.ones(shape, dtype=dtype)         # token ID 1, valid in any vocab
    if dtype == np.int8:
        return np.zeros(shape, dtype=dtype)        # zero is safe for int8 weights
    # float32: small non-zero avoids divide-by-zero in LayerNorm/BatchNorm
    return np.full(shape, 0.01, dtype=dtype)


def _run_with_timeout(label: str, fn, timeout_s: int, *args):
    """
    Run fn(*args) in a thread so a timeout can be imposed on blocking C++ calls
    like allocate_tensors() and invoke() that cannot be interrupted otherwise.
    """
    logger.info("    [tflite] %s starting (timeout: %ds)...", label, timeout_s)
    t0 = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args)
    try:
        result = future.result(timeout=timeout_s)
        logger.info("    [tflite] %s complete in %.2fs", label, time.monotonic() - t0)
        return result
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"{label} timed out after {timeout_s}s")
    finally:
        # wait=False releases our thread immediately without blocking for the worker.
        # The worker thread may continue running in the background until the OS
        # reclaims it, but our calling thread is unblocked.
        executor.shutdown(wait=False)


def validate_and_correct_pipeline(
    config: PipelineConfig,
    tflite_url: str,
) -> tuple[bool, str, bool, PipelineConfig]:
    """
    Downloads the TFLite model, runs dummy inference, and auto-corrects the
    pipeline config's input/output shapes and dtypes to match the actual model.

    Returns:
        (success, error_message, retryable, corrected_config)

        success:     True if validation passed.
        error_message: Human-readable failure reason, or "" on success.
        retryable:   True if the error is structural (wrong shape, wrong op, etc.)
                     and re-asking the LLM may produce a working config.
                     False for environment failures (timeout, model too large,
                     download error) where retrying with the same model is pointless.
        corrected_config: Auto-corrected config on success; original config on failure.
    """
    settings = get_settings()

    # Skip mediapipe_litert models — .task/.litertlm files can't be loaded by ai-edge-litert
    if config.metadata and config.metadata[0].framework == "mediapipe_litert":
        logger.info("  [validator] Skipping mediapipe_litert model — not loadable by ai-edge-litert")
        return True, "", False, config

    logger.info("  [validator] Starting validation")
    logger.debug("  [validator] URL: %s", tflite_url)

    temp_path = None
    try:
        # Step 1 — Download to temp file
        t_dl = time.monotonic()
        logger.debug("  [validator] Downloading model (limit: %d MB, timeout: %ds)...",
                     settings.MAX_VALIDATOR_DOWNLOAD_MB, settings.HF_FETCH_TIMEOUT_SECONDS)

        with httpx.Client(follow_redirects=True, timeout=settings.HF_FETCH_TIMEOUT_SECONDS) as client:
            with client.stream("GET", tflite_url) as r:
                r.raise_for_status()
                content_length = int(r.headers.get("content-length", 0))
                max_bytes = settings.MAX_VALIDATOR_DOWNLOAD_MB * 1024 * 1024
                if content_length > max_bytes:
                    msg = (f"Model too large to validate "
                           f"({content_length // 1024 // 1024} MB > {settings.MAX_VALIDATOR_DOWNLOAD_MB} MB)")
                    logger.warning("  [validator] %s", msg)
                    return False, msg, False, config  # not retryable — size is a hard limit
                if content_length:
                    logger.info("  [validator] Content-Length: %.1f MB", content_length / 1024 / 1024)
                else:
                    logger.warning("  [validator] No Content-Length header — chunked transfer, size unknown")

                download_deadline = time.monotonic() + settings.TFLITE_DOWNLOAD_TIMEOUT_SECONDS
                downloaded = 0
                last_progress_log = time.monotonic()

                with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
                    for chunk in r.iter_bytes():
                        if time.monotonic() > download_deadline:
                            raise TimeoutError(
                                f"Download timed out after {settings.TFLITE_DOWNLOAD_TIMEOUT_SECONDS}s "
                                f"({downloaded / 1024 / 1024:.1f} MB received)"
                            )
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(
                                f"Model exceeded size limit mid-download "
                                f"({downloaded // 1024 // 1024} MB > {settings.MAX_VALIDATOR_DOWNLOAD_MB} MB)"
                            )
                        now = time.monotonic()
                        if now - last_progress_log >= 5.0:
                            logger.info("  [validator] Downloading... %.1f MB received", downloaded / 1024 / 1024)
                            last_progress_log = now
                    temp_path = f.name

        actual_mb = os.path.getsize(temp_path) / 1024 / 1024
        logger.debug("  [validator] Download complete: %.2f MB in %.1fs", actual_mb, time.monotonic() - t_dl)

        # Step 2 — Load interpreter and inspect graph structure
        interpreter = Interpreter(model_path=temp_path)

        _run_with_timeout("allocate_tensors()", interpreter.allocate_tensors,
                          settings.TFLITE_INVOKE_TIMEOUT_SECONDS)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logger.info("  [validator] Graph: %d input(s), %d output(s)",
                    len(input_details), len(output_details))
        for d in input_details:
            logger.debug("    [validator] input  %-30s shape=%-20s dtype=%s",
                         d["name"], str(list(d["shape"])), d["dtype"].__name__)
        for d in output_details:
            logger.debug("    [validator] output %-30s shape=%-20s dtype=%s",
                         d["name"], str(list(d["shape"])), d["dtype"].__name__)

        if len(input_details) < len(config.inputs):
            msg = f"Model has {len(input_details)} inputs, pipeline declares {len(config.inputs)}"
            logger.warning("  [validator] %s", msg)
            return False, msg, True, config  # retryable — LLM declared too many inputs

        # Phase 1 — Auto-correct shapes and dtypes from graph metadata.
        # Available immediately after allocate_tensors(), before invoke().
        corrected = config.model_copy(deep=True)

        for i, detail in enumerate(input_details):
            if i < len(corrected.inputs):
                shape = [int(s) for s in detail["shape"]]
                old_shape = corrected.inputs[i].shape
                old_dtype = corrected.inputs[i].dtype
                if all(s > 0 for s in shape):
                    corrected.inputs[i].shape = shape
                corrected.inputs[i].dtype = _np_dtype_to_str(detail["dtype"])
                if old_shape != corrected.inputs[i].shape or old_dtype != corrected.inputs[i].dtype:
                    logger.debug("  [validator] corrected inputs[%d]: shape %s->%s  dtype %s->%s",
                                 i, old_shape, corrected.inputs[i].shape, old_dtype, corrected.inputs[i].dtype)

        for i, detail in enumerate(output_details):
            if i < len(corrected.outputs):
                shape = [int(s) for s in detail["shape"]]
                old_shape = corrected.outputs[i].shape
                old_dtype = corrected.outputs[i].dtype
                if all(s > 0 for s in shape):
                    corrected.outputs[i].shape = shape
                corrected.outputs[i].dtype = _np_dtype_to_str(detail["dtype"])
                if old_shape != corrected.outputs[i].shape or old_dtype != corrected.outputs[i].dtype:
                    logger.debug("  [validator] corrected outputs[%d]: shape %s->%s  dtype %s->%s",
                                 i, old_shape, corrected.outputs[i].shape, old_dtype, corrected.outputs[i].dtype)

        # Update resize_image step H/W from actual NHWC input shape
        if input_details and len(input_details[0]["shape"]) == 4:
            actual_h = int(input_details[0]["shape"][1])
            actual_w = int(input_details[0]["shape"][2])
            if actual_h > 0 and actual_w > 0:
                for block in corrected.preprocessing:
                    for step in block.steps:
                        if step.step == "resize_image":
                            if step.params.height != actual_h or step.params.width != actual_w:
                                logger.debug("  [validator] corrected resize_image: %dx%d -> %dx%d",
                                             step.params.height, step.params.width, actual_h, actual_w)
                            step.params.height = actual_h
                            step.params.width = actual_w

        # Phase 2 — Best-effort invoke to catch op-level runtime failures.
        # Shape correction is already done above; this only gates the return value.
        try:
            for detail in input_details:
                dummy = _make_dummy_input(detail["shape"], detail["dtype"])
                logger.debug("    [validator] set_tensor index=%-3d shape=%-20s dtype=%s",
                             detail["index"], str(list(detail["shape"])), detail["dtype"].__name__)
                interpreter.set_tensor(detail["index"], dummy)

            _run_with_timeout("invoke()", interpreter.invoke,
                              settings.TFLITE_INVOKE_TIMEOUT_SECONDS)
        except TimeoutError as e:
            msg = f"Model failed inference: {e}"
            logger.warning("  [validator] %s — not retryable", msg)
            # Return corrected (not config) so shape corrections survive in loose mode
            return False, msg, False, corrected
        except Exception as e:
            msg = f"Model failed inference: {e}"
            logger.warning("  [validator] %s — retryable", msg)
            return False, msg, True, corrected

        logger.info("  [validator] Validation passed ✓")
        return True, "", False, corrected

    except Exception as e:
        logger.error("  [validator] Unexpected error: %s", e, exc_info=True)
        return False, str(e), False, config  # unknown — don't retry

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug("  [validator] Cleaned up temp file")
