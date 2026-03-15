"""
Push TRIADS V13A to Hugging Face Hub.
Uploads: README.md, model_arch.py, triads_v13a_ensemble.pt
"""
import os
from huggingface_hub import HfApi

REPO_ID = "Rtx09/TRIADS"
UPLOAD_DIR = os.path.dirname(os.path.abspath(__file__))

FILES_TO_UPLOAD = [
    "README.md",
    "model_arch.py",
    "evaluate.py",
    "app.py",
    "triads_v13a_ensemble.pt",
]

def push():
    api = HfApi()
    
    # Verify all files exist
    for f in FILES_TO_UPLOAD:
        path = os.path.join(UPLOAD_DIR, f)
        if not os.path.exists(path):
            print(f"ERROR: Missing file: {path}")
            return
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ {f} ({size_mb:.1f} MB)")
    
    print(f"\nUploading to {REPO_ID}...")
    
    for f in FILES_TO_UPLOAD:
        path = os.path.join(UPLOAD_DIR, f)
        print(f"  Uploading {f}...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  ✓ {f} uploaded!")
    
    print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    push()
