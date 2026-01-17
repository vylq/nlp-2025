import argparse
import gzip
import pickle
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Text:
    label: str
    title: str
    text: str


def read_texts(fn):
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        for line in f:
            yield Text(*line.strip().split("\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/news.txt.gz", help="Путь к архиву (по умолчанию ./data/news.txt.gz)")
    ap.add_argument("--index-out", default="news.faiss", help="FAISS index file")
    ap.add_argument("--meta-out", default="news_meta.pkl", help="Pickle metadata")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    texts = []
    for i, t in enumerate(read_texts(args.input), start=1):
        texts.append(t)
        if args.max_docs and i >= args.max_docs:
            break

    model = SentenceTransformer("cointegrated/rubert-tiny2", device=args.device)

    docs = [(t.title + " " + t.text) for t in texts]

    emb = model.encode(
        docs,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, args.index_out)

    meta = [{"label": t.label, "title": t.title, "text": t.text} for t in texts]
    with open(args.meta_out, "wb") as f:
        pickle.dump(meta, f)

    print(f"OK: docs={len(meta)} dim={emb.shape[1]} -> {args.index_out}, {args.meta_out}")


if __name__ == "__main__":
    main()
