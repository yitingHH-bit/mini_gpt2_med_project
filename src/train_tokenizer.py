import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from utils import default_paths, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--vocab_size", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    paths = default_paths()
    text_path = Path(args.text) if args.text else (paths["data"] / "train.txt")
    out_dir = Path(args.out) if args.out else paths["tokenizer"]
    ensure_dir(out_dir)

    if not text_path.exists():
        raise FileNotFoundError(f"train text not found: {text_path}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(text_path)],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|sys|>", "<|usr|>", "<|asst|>"],
    )
    tokenizer.save_model(str(out_dir))
    print(f"✅ Tokenizer saved to: {out_dir}")

if __name__ == "__main__":
    main()
