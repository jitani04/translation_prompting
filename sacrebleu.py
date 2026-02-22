import json
import os
import sys
from datasets import load_dataset
import evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p not in ("", os.getcwd(), SCRIPT_DIR)]

def main():
    # Load first 10 test examples (en-amh)
    ds = load_dataset("masakhane/mafand", "en-amh", split="test")  # has fields "amh" and "en"
    ds10 = ds.select(range(10))

    def get_lang_text(example, lang):
        if "translation" in example and isinstance(example["translation"], dict):
            return example["translation"][lang]
        return example[lang]

    refs = [get_lang_text(ex, "en") for ex in ds10]   # reference translations
    srcs = [get_lang_text(ex, "amh") for ex in ds10]  # Amharic sources (for reporting)

    with open("fewshot.json", "r", encoding="utf-8") as f:
        preds = json.load(f)["preds"]

    assert len(preds) == 10, f"Expected 10 predictions, got {len(preds)}"

    sacrebleu = evaluate.load("sacrebleu")

    # sacrebleu expects references as list-of-lists (multiple refs per example)
    result = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])
    sample_scores = [
        sacrebleu.compute(predictions=[pred], references=[[ref]])["score"]
        for pred, ref in zip(preds, refs)
    ]
    best_idx = max(range(len(sample_scores)), key=lambda i: sample_scores[i])
    worst_idx = min(range(len(sample_scores)), key=lambda i: sample_scores[i])

    print("Average SacreBLEU over 10 examples:", result["score"])
    print("\n--- Pairs (src / pred / ref) ---")
    for i, (a, p, r) in enumerate(zip(srcs, preds, refs), start=1):
        print(f"\n{i}) AMH: {a}")
        print(f"   PRED: {p}")
        print(f"   REF : {r}")

    print("\n--- Best-scoring sample ---")
    print(f"Sample #{best_idx + 1} with SacreBLEU {sample_scores[best_idx]:.4f}")
    print(f"AMH : {srcs[best_idx]}")
    print(f"PRED: {preds[best_idx]}")
    print(f"REF : {refs[best_idx]}")

    print("\n--- Worst-scoring sample ---")
    print(f"Sample #{worst_idx + 1} with SacreBLEU {sample_scores[worst_idx]:.4f}")
    print(f"AMH : {srcs[worst_idx]}")
    print(f"PRED: {preds[worst_idx]}")
    print(f"REF : {refs[worst_idx]}")

if __name__ == "__main__":
    main()