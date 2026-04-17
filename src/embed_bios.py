import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import data_processed

def main():
    proc_path = os.path.join(data_processed, "processed.pt")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"run python src/data.py first; missing {proc_path}")

    proc = torch.load(proc_path, weights_only=False)
    texts = proc["texts"]
    print(f"loading sbert model for {len(texts):,} bios")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    out = os.path.join(data_processed, "bio_embeddings.npy")
    np.save(out, embeddings)
    print(f"saved {embeddings.shape} to {out}")

    proc["has_bio_embeddings"] = True
    torch.save(proc, proc_path)
    print(f"flagged processed.pt has_bio_embeddings = True")

if __name__ == "__main__":
    main()