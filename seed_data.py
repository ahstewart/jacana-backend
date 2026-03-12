from sqlmodel import Session, select
from database import engine
from schema import (
    UserDB, MLModelDB, ModelVersionDB, 
    PipelineConfig, PipelineStep, MLModelAsset, 
    ModelCategory, AssetType, DevicePlatform, InferenceLogDB
)
from datetime import datetime

def create_demo_data():
    with Session(engine) as session:
        # ==========================================
        # 1. Create a Developer (The User)
        # ==========================================
        dev_user = UserDB(
            username="jacana_architect",
            email="architect@jacana.app",
            is_developer=True
        )
        session.add(dev_user)
        session.commit()
        session.refresh(dev_user)
        
        print(f"Created User: {dev_user.username} (ID: {dev_user.id})")

        # ==========================================
        # 2. Create the Abstract Model (The Store Entry)
        # ==========================================
        water_thumper = MLModelDB(
            name="Watermelon Thumper",
            slug="watermelon-thumper",
            description_short="Detects ripeness via sound analysis.",
            category=ModelCategory.UTILITY,
            author_id=dev_user.id,
            tags=["audio", "agriculture", "fun"]
        )
        session.add(water_thumper)
        session.commit()
        session.refresh(water_thumper)

        # ==========================================
        # 3. Create the Logic (Pydantic Objects)
        # ==========================================
        # Define the Pipeline Logic strictly using Pydantic
        my_pipeline = PipelineConfig(
            input_nodes=["input_audio"],
            output_nodes=["ripeness_score"],
            pre_processing=[
                PipelineStep(step_name="normalize_audio", params={"sample_rate": 16000}),
                PipelineStep(step_name="spectrogram", params={"fft_size": 256})
            ],
            asset_map={"model_source": "tflite_file_v1"}
        )

        # Define the Assets strictly
        my_assets = [
            MLModelAsset(
                asset_key="tflite_file_v1",
                asset_type=AssetType.TFLITE,
                source_url="https://huggingface.co/jacana/melon/v1.tflite",
                file_size_bytes=102400,
                file_hash="sha256:abc123456...",
                is_hosted_by_us=False
            )
        ]

        # ==========================================
        # 4. Create the Version (Saving JSONB)
        # ==========================================
        # NOTE: We use .model_dump() so Postgres receives a Dict, not a Class
        version_1 = ModelVersionDB(
            model_id=water_thumper.id,
            version_string="1.0.0",
            
            # The Magic: Saving Pydantic structure into JSONB columns
            pipeline_spec=my_pipeline.model_dump(),
            assets=[a.model_dump() for a in my_assets],
            
            changelog="Initial release of the thumper."
        )
        session.add(version_1)
        session.commit()
        
        print(f"Created Version 1.0.0 for {water_thumper.name}")

def query_json_data():
    """
    Demonstrates the 'Diagnostic Hub' power: Querying INSIDE the JSON
    """
    with Session(engine) as session:
        print("\n--- Diagnostic Query ---")
        
        # SQL equivalent: 
        # SELECT * FROM model_versions WHERE pipeline_spec -> 'pre_processing' @> '[{"step_name": "spectrogram"}]'
        
        statement = select(ModelVersionDB).where(
            ModelVersionDB.pipeline_spec['pre_processing'].contains([{"step_name": "spectrogram"}])
        )
        results = session.exec(statement).all()
        
        for v in results:
            print(f"Found Model utilizing Spectrograms: Version {v.version_string} (ID: {v.id})")

if __name__ == "__main__":
    # create_demo_data() # Run this once
    query_json_data()    # Run this to test