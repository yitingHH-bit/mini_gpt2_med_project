import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils import default_paths

SYS_PROMPT = "Educational only. No diagnosis or treatment advice. Encourage seeing a clinician for personal concerns."


def format_prompt(user_text: str) -> str:
    return f"<|sys|> {SYS_PROMPT} <|usr|> {user_text} <|asst|>"

paths = default_paths()
DEFAULT_MODEL_DIR = str(paths["model"])

_tokenizer = None
_model = None
_device = None
_loaded_dir = None

def load_if_needed(model_dir: str):
    global _tokenizer, _model, _device, _loaded_dir
    if _model is None or _loaded_dir != model_dir:
        _tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        _model = GPT2LMHeadModel.from_pretrained(model_dir)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device).eval()
        _loaded_dir = model_dir

def gen(model_dir, prompt, max_new_tokens, temperature, top_p):
    load_if_needed(model_dir)
    full = format_prompt(prompt)

    max_ctx = int(getattr(_model.config, "n_positions", 256))

    # tokenize + truncate to context
    enc = _tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=max_ctx,
    )
    input_ids = enc["input_ids"].to(_device)

    # ensure we don't exceed context
    room = max(1, max_ctx - input_ids.shape[1])
    max_new_tokens = min(int(max_new_tokens), room)

    # (optional) debug: return token counts if you want
    # print(f"[debug] input_len={input_ids.shape[1]} room={room} max_new={max_new_tokens}")

    with torch.no_grad():
        out = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    text = _tokenizer.decode(out[0], skip_special_tokens=False)

    if "<|asst|>" in text:
        text = text.split("<|asst|>", 1)[-1]

    # clean special tokens that may remain
    for t in ["<|sys|>", "<|usr|>", "<|asst|>", "<|pad|>", "<|bos|>", "<|eos|>"]:
        text = text.replace(t, "")

    return text.strip()



with gr.Blocks(title="Mini Medical GPT-2 Demo") as demo:
    gr.Markdown("# Mini Medical GPT-2 (Learning Demo)\nEducation/research only. Not for diagnosis/treatment.")
    model_dir = gr.Textbox(label="Model directory", value=DEFAULT_MODEL_DIR)
    prompt = gr.Textbox(label="Prompt", value="Explain what a biopsy is.", lines=3)
    with gr.Row():
        max_new_tokens = gr.Slider(16, 256, value=120, step=1, label="Max new tokens")
        temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
    btn = gr.Button("Generate")
    out = gr.Textbox(label="Output", lines=10)
    btn.click(gen, [model_dir, prompt, max_new_tokens, temperature, top_p], out)

if __name__ == "__main__":
    demo.launch()
