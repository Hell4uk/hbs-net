# HBS-NET

Hybrid Belief-State Network: a contradiction-aware hybrid architecture for
compact language models. Pure PyTorch implementation targeting a single
RTX 5060 8GB (also runs on CPU).

## Architecture

```
Embedding
  -> n_ssm_layers x MambaBlock (selective SSM, belief state h)
  -> DialecticalGate (contradiction index c, alpha = beta + (1-beta)*c)
  -> n_attention_layers x AttentionBlock (precise retrieval)
  -> final norm -> tied LM head
```

- **SelectiveSSM**: diagonal SSM with input-dependent B, C, dt; causal
  depthwise conv; chunked parallel scan for training and an exact
  recurrent step mode for generation.
- **DialecticalGate**: computes a per-token contradiction index from the
  belief state and modulates how strongly new evidence updates the state.
- **Composite loss**: `L = L_text + aux_weight * L_contradiction +
energy_weight * E`, where `E = ||h||^2` is the state energy.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
```

## Train

```powershell
.\.venv\Scripts\python -m src.scripts.train --config config/base.yaml --epochs 50
```

Options: `--device cpu|cuda`, `--n-samples`, `--n-corpus`, `--epochs`.
The tokenizer is trained automatically on the built-in synthetic Russian
corpus and saved to `data/tokenizer.json`; checkpoints go to
`checkpoints/hbs_final.pt`.

## Generate

```powershell
.\.venv\Scripts\python -m src.scripts.generate `
  --checkpoint checkpoints/hbs_final.pt `
  --question "Какая трава?" `
  --evidence "трава была зелёная." "трава оказалась жёлтая." `
  --temperature 0.8 --max-new-tokens 32
```

Prints the answer and the mean contradiction index over generated tokens.

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

`tests/test_model.py` includes a step/sequence consistency test that
verifies recurrent generation exactly reproduces teacher-forced forward
passes.

## Project layout

- `src/hbs/tokenizer` — BPE tokenizer and synthetic corpus
- `src/hbs/model` — Mamba blocks, attention, dialectical gate, HBSNet
- `src/hbs/dataset` — synthetic contradiction dataset and collate
- `src/hbs/training` — composite loss and training loop
- `src/hbs/inference` — autoregressive generator
- `src/scripts` — CLI entry points (`train`, `generate`)
- `config/base.yaml` — model and training hyperparameters
