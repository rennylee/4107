# test_imports.py
print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
    
    from sentence_transformers import SentenceTransformer
    print(f"✓ Sentence-Transformers imported")
    
    import huggingface_hub
    print(f"✓ Huggingface Hub {huggingface_hub.__version__}")
    
    print("\nAll imports successful!")
except Exception as e:
    print(f"❌ Error: {e}")