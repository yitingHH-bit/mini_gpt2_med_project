import os
from pathlib import Path

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def default_paths():
    root = project_root()
    return {
        "data": root / "data",
        "outputs": root / "outputs",
        "tokenizer": root / "outputs" / "tokenizer_1k",
        "model": root / "outputs" / "mini_gpt2_med",
    }

def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def clean_med_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()

def chunk_by_chars(texts, max_chars=500):
    out, buf = [], ""
    for t in texts:
        if not t:
            continue
        if len(buf) + len(t) + 1 <= max_chars:
            buf = (buf + " " + t).strip()
        else:
            if buf:
                out.append(buf)
            buf = t
    if buf:
        out.append(buf)
    return out
