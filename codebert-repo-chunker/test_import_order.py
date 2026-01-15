
print("Testing FAISS then Torch...")
try:
    import faiss
    print("FAISS imported.")
    import torch
    print("Torch imported.")
    # Try to use torch
    x = torch.tensor([1.0])
    print("Torch tensor created.")
except Exception as e:
    print(f"Error: {e}")
