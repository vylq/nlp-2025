import argparse
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="FAISS index file")
    ap.add_argument("--meta", required=True, help="Pickle metadata")
    ap.add_argument("query", help="Поисковой запрос")
    ap.add_argument("-k", type=int, default=5, help="Top-K")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--min-score", type=float, default=-1.0, help="Минимальный порог")
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)

    model = SentenceTransformer("cointegrated/rubert-tiny2", device=args.device)
    q = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    overshoot = max(args.k * 10, args.k)
    scores, ids = index.search(q, overshoot)

    found = 0
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        m = meta[idx]
        if score < args.min_score:
            continue

        found += 1
        title = m["title"].replace("\n", " ")
        snippet = m["text"].replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        print(f"[{found}] score={score:.4f}\n{title}\n{snippet}\n")
        if found >= args.k:
            break

    if found == 0:
        print("Нет результатов")


if __name__ == "__main__":
    main()
