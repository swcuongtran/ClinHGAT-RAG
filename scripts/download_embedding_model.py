from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = PROJECT_ROOT / "local_models" / "vietnamese-sbert"

print("Downloading and packaging embedding model...")
print(f"Model directory: {LOCAL_MODEL_DIR}")

model = SentenceTransformer("keepitreal/vietnamese-sbert")
model.save(str(LOCAL_MODEL_DIR))

print(f"Saved embedding model to {LOCAL_MODEL_DIR}")
