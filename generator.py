import os
import tempfile
import httpx
from pydantic import ValidationError, BaseModel, Field
from typing import Optional
from sqlmodel import Session, select
from huggingface_hub import HfApi
from google import genai
from google.genai import types
from tflite_support import metadata 

# Database and Schema import
from database import engine
from schema import MLModelDB, ModelVersionDB
from pipeline_schema import PipelineConfig
from config import get_settings

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
    try:
        # Download the raw README.md file from the specific commit
        url = f"https://huggingface.co/{repo_id}/resolve/{commit_sha}/README.md"
        response = httpx.get(url, timeout=10.0, follow_redirects=True)
        
        if response.status_code == 200:
            return response.text
        return "No README found."
    except Exception as e:
        print(f"Warning: Failed to fetch README for {repo_id}: {e}")
        return "Failed to fetch README."

def fetch_hf_model_card(repo_id: str) -> str:
    """
    Fetches the structured Model Card metadata (YAML frontmatter and tags) 
    from the Hugging Face Hub API.
    """
    api = HfApi()
    try:
        info = api.model_info(repo_id=repo_id)
        # cardData contains the structured YAML frontmatter of the model card
        card_data = getattr(info, 'cardData', {})
        return str(card_data) if card_data else "No structured Model Card data found."
    except Exception as e:
        print(f"Warning: Failed to fetch Model Card for {repo_id}: {e}")
        return "Failed to fetch Model Card data."

def fetch_tflite_metadata(tflite_url: str) -> str:
    """
    Attempts to download the TFLite file and extract embedded FlatBuffer metadata.
    """
    try:
        # We only download the model to a temporary file just for this extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as tmp_file:
            print(f"  -> Downloading TFLite binary for metadata extraction...")
            with httpx.stream("GET", tflite_url, follow_redirects=True) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=8192):
                    tmp_file.write(chunk)
            tmp_path = tmp_file.name

        # Extract metadata
        displayer = metadata.MetadataDisplayer.with_model_file(tmp_path)
        metadata_text = displayer.get_metadata_json()
        
        # Cleanup temp file
        os.remove(tmp_path)
        
        return metadata_text if metadata_text else "No embedded metadata found in TFLite file."
    
    except Exception as e:
        print(f"  -> Warning: Failed to extract TFLite metadata: {e}")
        return "Metadata extraction failed."

def generate_pipeline_config(task: str, readme_text: str, metadata_text: str, model_card_text: str) -> PipelineGenerationResult | None:
    """
    Feeds the unstructured text to Gemini and forces it to return 
    a strict JSON object matching our PipelineGenerationResult wrapper.
    """
    system_prompt = f"""
    You are an expert Edge AI engineer responsible for writing metadata files for AI models. These files will be used to run inference on these models on edge devices. 
    Your job is to analyze the following Hugging Face Model Card, README, and TFLite Metadata for the AI model, and generate a strict configuration file for mobile deployment.
    
    Model Task: {task}
    
    CRITICAL INSTRUCTION:
    It is likely you will be given models for tasks we do not yet support (like Audio, NLP) or models requiring complex custom preprocessing not defined in our schema.
    If the model cannot be PERFECTLY configured using our exact schema, you MUST set 'is_supported' to false, briefly explain why in 'reasoning', and leave 'config' null.
    
    Instructions if supported:
    1. TENSORS: explicitly define 'inputs' and 'outputs' using exact tensor names, shapes, and dtypes.
    2. ROUTING: 'input_name' and 'source_tensors' must match the defined tensors.
    3. PREPROCESSING: Create a LIST of PreprocessBlock objects. Each block has:
       - input_name: str (name of the input tensor)
       - expects_type: Literal["image", "audio", "text"]
       - steps: List of PreprocessStep objects
    4. POSTPROCESSING: Create a LIST of PostprocessBlock objects. Each block has:
       - output_name: str (logical name for the output)
       - interpretation: str (e.g., 'classification_scores')
       - source_tensors: List of output tensor names
       - coordinate_format: Optional[str] (for detection models)
       - steps: List of PostprocessStep objects
    5. You MUST map requirements ONLY to the exact structures and allowed literals defined in the schema.
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

    try:
        # Using Gemini's structured outputs with our new wrapper schema
        response = client.models.generate_content(
            model=settings.PIPELINE_GENERATION_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_json_schema=PipelineGenerationResult.model_json_schema(),
            ),
            contents=user_prompt, 
        )
        
        # Parse into the wrapper object and return the whole result so we can save the reasoning
        result = PipelineGenerationResult.model_validate_json(response.text)
        return result

    except Exception as e:
        print(f"  -> LLM Generation Failed: {e}")
        return None
    

def run_generator_for_version(version: ModelVersionDB, model: MLModelDB, session: Session) -> bool:
    """
    Executes the pipeline generation for a single version and commits to DB.
    Returns True if generation was successfully processed (even if unsupported), False if it failed.
    """
    # 1. Gather Context
    readme_text = fetch_hf_readme(model.hf_model_id, version.commit_sha)
    model_card_text = fetch_hf_model_card(model.hf_model_id)
    
    # Extract the TFLite URL from our strictly validated dictionary
    tflite_url = version.assets.get("tflite") 
    metadata_text = fetch_tflite_metadata(tflite_url) if tflite_url else "No TFLite URL found."

    # 2. Ask the LLM to generate the configuration
    print(f"  -> Sending context to LLM (Task: {model.task})...")
    result = generate_pipeline_config(model.task, readme_text, metadata_text, model_card_text)

    # 3. Handle the Result
    if result:
        if result.is_supported and result.config:
            print(f"  -> Success! Generated config with {len(result.config.preprocessing)} preprocessing steps.")
            
            # Convert the Pydantic model back to a dictionary for Postgres JSONB storage
            version.pipeline_spec = result.config.model_dump(mode='json')
            version.status = "configured"
            version.is_supported = True
            version.unsupported_reason = None
        else:
            print(f"  -> Model gracefully rejected by LLM. Reason: {result.reasoning}")
            
            # Store the rejection metadata
            version.status = "unsupported"
            version.is_supported = False
            version.unsupported_reason = result.reasoning
        
        # Commit the changes to the database
        session.add(version)
        session.commit()
        return True
    else:
        print(f"  -> Failed to generate a valid response. Leaving as 'unconfigured'.")
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


def process_all_unconfigured():
    """
    The main job loop. Finds all unconfigured versions, gathers context, 
    asks the LLM for a config, and saves it if successful.
    """
    print("Starting LLM Pipeline Generator...")
    
    with Session(engine) as session:
        statement = select(ModelVersionDB, MLModelDB).join(MLModelDB).where(ModelVersionDB.status == "unconfigured")
        results = session.exec(statement).all()

        if not results:
            print("No unconfigured models found. It's a miracle!")
            return

        for version, model in results:
            print(f"\nProcessing: {model.name} (Version: {version.version_name})")
            
            # 1. Gather Context
            readme_text = fetch_hf_readme(model.hf_model_id, version.commit_sha)
            model_card_text = fetch_hf_model_card(model.hf_model_id)
            
            # Extract the TFLite URL from our strictly validated dictionary
            tflite_url = version.assets.get("tflite") 
            metadata_text = fetch_tflite_metadata(tflite_url) if tflite_url else "No TFLite URL found."

            # 2. Ask the LLM to generate the configuration
            print(f"  -> Sending context to LLM (Task: {model.task})...")
            result = generate_pipeline_config(model.task, readme_text, metadata_text, model_card_text)

            # 3. Handle the Result
            if result:
                if result.is_supported and result.config:
                    print(f"  -> Success! Generated config with {len(result.config.preprocessing)} preprocessing steps and {len(result.config.postprocessing)} postprocessing steps.")
                    
                    # Convert the Pydantic model back to a dictionary for Postgres JSONB storage
                    version.pipeline_spec = result.config.model_dump(mode='json')
                    version.status = "configured"
                    version.is_supported = True
                    version.unsupported_reason = None
                else:
                    print(f"  -> Model gracefully rejected by LLM. Reason: {result.reasoning}")
                    
                    # Store the rejection metadata
                    version.status = "unsupported"
                    version.is_supported = False
                    version.unsupported_reason = result.reasoning
                
                # Commit the changes to the database
                session.add(version)
                session.commit()
            else:
                print(f"  -> Failed to generate a valid response. Leaving as 'unconfigured'.")


if __name__ == "__main__":
    #process_all_unconfigured()
    run_generator_for_huggingface_model(repo_id="byoussef/MobileNetV4_Conv_Small_TFLite_224", commit_sha="ca1cd2705679b77b6e7008e67161d1276913bfc3")