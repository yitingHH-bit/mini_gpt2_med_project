import argparse, math
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils import default_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--eval_txt", type=str, default=None)
    ap.add_argument("--max_lines", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=64)
    args = ap.parse_args()

    paths = default_paths()
    model_dir = args.model_dir or str(paths["model"])
    eval_txt = args.eval_txt or str(paths["data"] / "train.txt")

    tok = GPT2TokenizerFast.from_pretrained(model_dir)
    mdl = GPT2LMHeadModel.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    losses = []
    with open(eval_txt, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= args.max_lines:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            enc = tok(line, truncation=True, max_length=args.seq_len, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = mdl(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            losses.append(out.loss.item())

    if not losses:
        raise SystemExit("No valid eval lines found.")
    mean_loss = sum(losses)/len(losses)
    ppl = math.exp(mean_loss)
    print(f"lines={len(losses)} loss={mean_loss:.4f} ppl={ppl:.2f}")

if __name__ == "__main__":
    main()
