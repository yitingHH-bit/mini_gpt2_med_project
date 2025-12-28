import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils import default_paths

SYS_PROMPT = "Educational only. No diagnosis or treatment advice. Encourage seeing a clinician for personal concerns."


def format_prompt(user_text: str) -> str:
    return f"<|sys|> {SYS_PROMPT} <|usr|> {user_text} <|asst|>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--prompt", type=str, default="Explain what screening means for cancer.")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    paths = default_paths()
    model_dir = args.model_dir or str(paths["model"])
    tok = GPT2TokenizerFast.from_pretrained(model_dir)
    mdl = GPT2LMHeadModel.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    prompt = format_prompt(args.prompt)
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    if "<|asst|>" in text:
        text = text.split("<|asst|>", 1)[-1].strip()
    print(text)

if __name__ == "__main__":
    main()
