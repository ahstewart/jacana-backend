import yaml
import httpx
from huggingface_hub import HfApi
from typing import Dict, Any, Optional

SAFE_LICENSES = {"apache-2.0", "mit", "bsd", "bsd-3-clause", "cc0"}
RESTRICTED_LICENSES = {"gpl", "gpl-2.0", "gpl-3.0", "agpl", "cc-by-nc", "cc-by-nc-sa"}

def scan_hf_repo_for_version_assets(repo_id: str, commit_sha: str, license_type: str) -> Optional[Dict[str, Any]]:
    """
    Pocket AI Strategy 1 & 2 Combined.
    Scans for a pocket_ai.yaml config and dynamically builds the AssetPointers map.
    If no config is found, falls back to the Heuristic Scanner.
    """
    api = HfApi()
    
    try:
        model_info = api.model_info(repo_id=repo_id, revision=commit_sha, files_metadata=True)
    except Exception as e:
        print(f"Failed to fetch repo info for {repo_id} at {commit_sha}: {e}")
        return None

    tflite_files = []
    auxiliary_files = []
    config_file = None
    
    # 1. Parse the File Tree for Assets AND our Standardized Config
    for file in model_info.siblings:
        filename = file.rfilename.lower()
        if filename.endswith(".tflite"):
            tflite_files.append(file)
        elif filename == "pocket_ai.yaml":
            config_file = file
        elif any(ext in filename for ext in [".txt", ".json"]):
            auxiliary_files.append(file)

    if not tflite_files:
        return None

    # Apply Heuristics (Pick the best TFLite file, ignore EdgeTPU by default)
    best_tflite = next((f for f in tflite_files if "edgetpu" not in f.rfilename.lower()), tflite_files[0])

    base_resolve_url = f"https://huggingface.co/{repo_id}/resolve/{commit_sha}"

    # --- THE NEW JSONB ASSET POINTERS MAP ---
    # This matches the schema.py AssetPointers Pydantic model exactly.
    assets = {
        "tflite": f"{base_resolve_url}/{best_tflite.rfilename}",
        "labels": None,
        "tokenizer": None,
        "vocab": None,
        "anchors": None
    }

    # Intelligently map auxiliary files
    for f in auxiliary_files:
        lower_name = f.rfilename.lower()
        url = f"{base_resolve_url}/{f.rfilename}"
        
        if "tokenizer.json" in lower_name:
            assets["tokenizer"] = url
        elif "vocab.txt" in lower_name:
            assets["vocab"] = url
        elif "anchor" in lower_name:
            assets["anchors"] = url
        elif any(x in lower_name for x in ["label", "class", "synset"]):
            if not assets["labels"]:  # Prefer the first one found if multiple exist
                assets["labels"] = url

    clean_license = (license_type or "unknown").lower()

    # --- STRATEGY 2: THE STANDARDIZED CONFIG INGESTION ---
    pipeline_spec = None
    status = "unconfigured"

    if config_file:
        try:
            # We ONLY download this tiny text file, bypassing the massive binaries
            raw_yaml_url = f"{base_resolve_url}/{config_file.rfilename}"
            response = httpx.get(raw_yaml_url, timeout=5.0)
            response.raise_for_status()
            
            pipeline_spec = yaml.safe_load(response.text)
            status = "configured" # It has our spec, it's ready for edge execution!
            
        except Exception as e:
            print(f"Found pocket_ai.yaml in {repo_id} but failed to parse: {e}")
            status = "unconfigured"

    # Build the Version Payload
    return {
        "version_name": commit_sha[:7],
        "commit_sha": commit_sha,
        "is_hosted_by_us": False,
        "assets": assets, # Passes our strict JSONB Pydantic dictionary
        "pipeline_spec": pipeline_spec,
        "license_type": clean_license,
        "is_commercial_safe": clean_license in SAFE_LICENSES,
        "requires_commercial_warning": clean_license in RESTRICTED_LICENSES,
        "file_size_bytes": best_tflite.size or 0,
        "status": status 
    }