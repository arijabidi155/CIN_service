from huggingface_hub import snapshot_download
import os

print("Starting model download...")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

snapshot_download(
    repo_id="BesirVelioglu/id-card-detection-rd-report",
    local_dir=models_dir
)
print("Model download complete!")
