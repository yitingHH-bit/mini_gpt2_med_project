import argparse
import json
from pathlib import Path
from utils import ensure_dir, clean_med_text, chunk_by_chars

def read_txt(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
    return paras

def read_csv(path: Path, text_col: str):
    import csv
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if text_col not in (reader.fieldnames or []):
            raise ValueError(f"CSV missing column '{text_col}'. Found: {reader.fieldnames}")
        for r in reader:
            rows.append(r.get(text_col, "") or "")
    return rows

def read_jsonl(path: Path, text_key: str):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj.get(text_key, "") or "")
    return rows

def read_hf(dataset_name: str, split: str, fields: list[str]):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    texts = []
    for ex in ds:
        parts = []
        for k in fields:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list):
                parts.extend([str(x).strip() for x in v if str(x).strip()])
        if parts:
            texts.append(" \n".join(parts))
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_txt", type=str, default=None)
    ap.add_argument("--input_csv", type=str, default=None)
    ap.add_argument("--input_jsonl", type=str, default=None)
    ap.add_argument("--text_col", type=str, default="abstract")
    ap.add_argument("--text_key", type=str, default="abstract")
    ap.add_argument("--hf_dataset", type=str, default=None)
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--hf_fields", type=str, default="context,long_answer,question")
    ap.add_argument("--out_txt", type=str, required=True)
    ap.add_argument("--max_chars", type=int, default=500)
    ap.add_argument("--min_len", type=int, default=40)
    args = ap.parse_args()

    sources = [args.input_txt, args.input_csv, args.input_jsonl, args.hf_dataset]
    if sum(x is not None for x in sources) != 1:
        raise SystemExit("Choose exactly ONE source: --input_txt OR --input_csv OR --input_jsonl OR --hf_dataset")

    if args.input_txt:
        texts = read_txt(Path(args.input_txt))
    elif args.input_csv:
        texts = read_csv(Path(args.input_csv), args.text_col)
    elif args.input_jsonl:
        texts = read_jsonl(Path(args.input_jsonl), args.text_key)
    else:
        fields = [f.strip() for f in args.hf_fields.split(",") if f.strip()]
        texts = read_hf(args.hf_dataset, args.hf_split, fields)

    cleaned = [clean_med_text(str(t)) for t in texts]
    cleaned = [t for t in cleaned if len(t) >= args.min_len]
    samples = chunk_by_chars(cleaned, max_chars=args.max_chars)

    out_path = Path(args.out_txt)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.replace("\n", " ").strip() + "\n")

    print(f"✅ Wrote corpus: {out_path} (samples={len(samples)})")
    if samples:
        print("Sample:", samples[0][:300])

if __name__ == "__main__":
    main()
