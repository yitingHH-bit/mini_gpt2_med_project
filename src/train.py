import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments
)

from utils import default_paths, ensure_dir, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_text_dataset(txt_path: Path) -> Dataset:
    lines = []
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            lines.append(ln)
    return Dataset.from_dict({"text": lines})


def mask_labels_after_asst(tokenizer: GPT2TokenizerFast, input_ids: torch.Tensor) -> torch.Tensor:
    """
    labels default = input_ids
    ignore loss (set -100) for everything up to and including the first <|asst|> token.
    """
    labels = input_ids.clone()

    asst_id = tokenizer.convert_tokens_to_ids("<|asst|>")
    if asst_id is None or asst_id < 0:
        # if somehow token id is not found, ignore all
        return torch.full_like(labels, -100)

    for i in range(labels.size(0)):
        row = input_ids[i]
        idx = (row == asst_id).nonzero(as_tuple=False)
        if idx.numel() == 0:
            # no <|asst|> => don't train on this line
            labels[i, :] = -100
        else:
            start = int(idx[0].item()) + 1  # start AFTER <|asst|>
            labels[i, :start] = -100
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", type=str, default=None)
    ap.add_argument("--tokenizer_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=256)   # <-- recommend 256 for your use
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    paths = default_paths()
    train_txt = Path(args.train_txt) if args.train_txt else (paths["data"] / "train.txt")
    tok_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else paths["tokenizer"]
    out_dir = Path(args.out_dir) if args.out_dir else paths["model"]
    ensure_dir(out_dir)

    tokenizer = GPT2TokenizerFast(
        vocab_file=str(tok_dir / "vocab.json"),
        merges_file=str(tok_dir / "merges.txt"),
    )
    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "additional_special_tokens": ["<|sys|>", "<|usr|>", "<|asst|>"],
    })

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.seq_len,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_inner=128,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    ds = load_text_dataset(train_txt)

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )

    tokenized = ds.map(tok, batched=True, remove_columns=["text"])

    # ✅ custom collator that masks labels before <|asst|>
    def collator(features):
        batch = tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = mask_labels_after_asst(tokenizer, batch["input_ids"])
        return batch

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"✅ Model saved to: {out_dir}")


if __name__ == "__main__":
    main()
