import sys
import os
from dotenv import load_dotenv

load_dotenv()

def check_env():
    print("=" * 50)
    print("PatentRank Environment & Hardware Verification")
    print("=" * 50)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    # PyTorch & Hardware
    import torch
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        print("  -> Running in CPU mode (as expected per BUILD_PLAN.md - no local GPU required).")
    else:
        print(f"  -> CUDA device: {torch.cuda.get_device_name(0)}")

    # Google GenAI SDK & API Key verification
    print("\nGoogle AI Studio (Gemini API) Verification:")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "" or "your_gemini_api_key" in api_key:
        print("  [!] GEMINI_API_KEY not configured or using placeholder.")
        print("      Please add your key to .env: GEMINI_API_KEY=your_key")
        return

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents="PatentRank test embedding"
        )
        emb = response.embeddings[0].values
        print(f"  [OK] Successfully connected to Gemini API!")
        print(f"  [OK] gemini-embedding-2 returned embedding with dimension: {len(emb)}")
    except Exception as e:
        print(f"  [!] Failed to call Gemini API: {e}")

if __name__ == "__main__":
    check_env()
