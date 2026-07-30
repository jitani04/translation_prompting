# Low-Resource Machine Translation via LLM Prompting

An experiment in **Amharic → English** machine translation using large language
models under different in-context prompting strategies, evaluated with SacreBLEU.
Amharic is a low-resource language, which makes it a challenging benchmark for
zero- and few-shot translation.

## Approach

Three prompting regimes are compared, each producing translations for the same
10 held-out test sentences:

- `zeroshot.json` — zero-shot (instruction only, no examples)
- `oneshot.json` — one in-context example
- `fewshot.json` — several in-context examples

Predictions are scored against the reference translations from the
[MAFAND](https://huggingface.co/datasets/masakhane/mafand) `en-amh` test split.

## Evaluation

`sacrebleu.py` loads the MAFAND test set, reads a set of model predictions, and
reports the average SacreBLEU score along with the best- and worst-scoring
sentences, using Hugging Face `evaluate`.

```bash
pip install datasets evaluate sacrebleu
python sacrebleu.py     # scores fewshot.json by default
```

## Tech stack

Python, Hugging Face `datasets` & `evaluate`, SacreBLEU, MAFAND (Masakhane) corpus.
