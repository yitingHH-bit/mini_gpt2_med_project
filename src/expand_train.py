import argparse, random, re
from pathlib import Path

SYS = "Cancer risk education only. No diagnosis or treatment advice. For personal concerns, consult a licensed clinician."

QUESTION_TEMPLATES = [
    "{q}",
    "In simple terms, {q}",
    "Can you explain: {q}",
    "As a quick overview, {q}",
    "What should I know about this: {q}",
]

ANSWER_PREFIX = ["", "In general, ", "In plain language, ", "A helpful way to think about it is: "]
ANSWER_SUFFIX = ["", " If you have personal symptoms or concerns, consult a licensed clinician."]

def normalize(s: str) -> str:
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_line(line: str):
    # Expect: <|sys|>...<|usr|>Q<|asst|>A
    if "<|usr|>" not in line or "<|asst|>" not in line:
        return None
    q = line.split("<|usr|>", 1)[1].split("<|asst|>", 1)[0]
    a = line.split("<|asst|>", 1)[1]
    return normalize(q), normalize(a)

def bullets_from_sentences(a: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", a.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return a
    parts = parts[:3]
    return "Key points: " + " | ".join(parts)

def make_variants(q: str, a: str):
    out = []
    for qt in QUESTION_TEMPLATES:
        qv = normalize(qt.format(q=q))
        av1 = normalize(random.choice(ANSWER_PREFIX) + a + random.choice(ANSWER_SUFFIX))
        av2 = normalize(bullets_from_sentences(a) + random.choice(ANSWER_SUFFIX))
        av3 = normalize("Short answer: " + a.split(".")[0].strip() + "." + random.choice(ANSWER_SUFFIX))
        for av in (av1, av2, av3):
            out.append(f"<|sys|>{SYS}<|usr|>{qv}<|asst|>{av}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", type=str, default="data/train.txt")
    ap.add_argument("--out_txt", type=str, default="data/train_aug.txt")
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    in_path = Path(args.in_txt)
    out_path = Path(args.out_txt)

    base = []
    for ln in in_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parsed = parse_line(ln)
        if parsed:
            base.append(parsed)

    if not base:
        raise RuntimeError("No valid <|usr|>...<|asst|> lines found in input file.")

    lines = []
    for q, a in base:
        lines.extend(make_variants(q, a))

    random.shuffle(lines)
    while len(lines) < args.target:
        q, a = random.choice(base)
        lines.extend(make_variants(q, a))
        random.shuffle(lines)

    lines = lines[:args.target]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ wrote {len(lines)} lines -> {out_path}")

if __name__ == "__main__":
    main()
