# nanoGPT

A small character-level GPT training project built with PyTorch. This repo is set up for training on a plain text corpus, saving a checkpoint, and generating text from the trained model.

## Features

- Character-level tokenizer built directly from your dataset
- Lightweight GPT-style transformer model
- YAML-based training configuration
- Simple training and text generation scripts
- Support for external datasets through CLI flags

## Requirements

- Python 3.10+ recommended
- `torch`
- `pyyaml`
- `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
nanoGPT/
├── checkpoints/        # Saved model checkpoints
├── configs/            # YAML config files
├── data/               # Default sample dataset
├── scripts/            # Train and generate entry points
├── src/
│   ├── inference/      # Inference helpers
│   ├── models/         # Transformer model components
│   ├── training/       # Batch sampling and training loop
│   └── utils/          # Config loading, tokenizer, metrics
└── tests/              # Test directory
```

## Default Training

The default training script reads `data/input.txt` and uses `configs/train.yaml`.

Run training:

```bash
python3 scripts/train.py
```

This writes the trained model to `checkpoints/model.pt`.

## Generate Text

After training, generate text from the saved checkpoint:

```bash
python3 scripts/generate.py
```

## Use an External Dataset

You can point the scripts at any plaintext dataset file.

Train with an external dataset:

```bash
python3 scripts/train.py --data-path "/absolute/path/to/dataset.txt"
```

Generate with the same dataset vocabulary:

```bash
python3 scripts/generate.py \
  --data-path "/absolute/path/to/dataset.txt" \
  --checkpoint checkpoints/model.pt \
  --tokens 300
```

Important: the tokenizer is built from the dataset text itself. That means generation should use the same dataset, or at least a dataset with the same character set, as the one used during training.

## Configuration

Training defaults live in `configs/train.yaml`.

Current defaults:

```yaml
batch_size: 32
block_size: 64
embed_dim: 128
num_heads: 4
num_layers: 2
dropout: 0.1
learning_rate: 0.0003
epochs: 10
```

You can pass a different config file with:

```bash
python3 scripts/train.py --config configs/train.yaml
python3 scripts/generate.py --config configs/train.yaml
```

## Notes

- Datasets must be plain text files.
- This is a character-level model, so quality depends heavily on dataset size and cleanliness.
- Checkpoints are overwritten at `checkpoints/model.pt` unless you change the script behavior.

## Next Steps

Good follow-up improvements for this repo would be:

- Add a validation split and evaluation loss tracking
- Save tokenizer metadata alongside checkpoints
- Add tests for training and generation entry points
- Support alternative tokenizers beyond character-level encoding
