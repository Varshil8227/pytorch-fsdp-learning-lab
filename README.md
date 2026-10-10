# PyTorch FSDP Learning Lab

A small, hands-on repository for learning the **PyTorch Fully Sharded Data Parallel (FSDP) API** with three educational training examples.

> **Learning project only:** These scripts are small demonstrations for understanding training loops, GPU execution, metrics, checkpoints, and FSDP setup. They do **not** train an original, production-ready, or general-purpose AI model. The LSTM example is a tiny character-level language model trained on a repeated toy text. A single-GPU run also does not demonstrate the memory-saving and scaling benefits of multi-GPU sharding.

## Learning goals

- Check whether PyTorch can access an NVIDIA GPU.
- Understand a basic PyTorch training loop.
- Measure training loss, test/validation metrics, and elapsed training time.
- Wrap a model with `FullyShardedDataParallel` (FSDP).
- Save a model state dictionary.
- Compare three educational tasks: tabular classification, image classification, and next-character prediction.

## Repository structure

```text
pytorch-fsdp-learning-lab/
├── README.md
├── .gitignore
├── requirements.txt
├── 01_ml_fsdp.py
├── 02_cnn_fsdp.py
├── 03_lstm_fsdp.py
├── data/                  # Downloaded/generated data; not committed
├── checkpoints/           # Saved model files; not committed
└── logs/                  # Local logs; not committed
```

The `data/`, `checkpoints/`, and `logs/` directories are created locally when needed. Git does not track empty directories, and that is expected.

## Examples

### 1. `01_ml_fsdp.py` — Small tabular classifier

**What it does**

- Creates a synthetic binary-classification dataset with scikit-learn.
- Splits the data into training and test sets.
- Scales features using a scaler fitted only on the training set.
- Trains a small multilayer perceptron (MLP) using PyTorch.
- Prints epoch loss, test accuracy, training time, and the detected device.
- Saves model weights to `checkpoints/ml/model.pt`.

**What to learn**

- The structure of a basic PyTorch training loop.
- How tensors and batches are moved to a CUDA device.
- The roles of `CrossEntropyLoss`, `backward()`, and `optimizer.step()`.
- How FSDP wraps a PyTorch model and how to request a full model state dictionary.

**Important note:** This is a small neural-network classifier on synthetic data. It is not a traditional scikit-learn estimator trained by FSDP; ordinary scikit-learn estimators are not wrapped by PyTorch FSDP.

### 2. `02_cnn_fsdp.py` — MNIST image classifier

**What it does**

- Downloads the MNIST handwritten-digit dataset if it is not already available.
- Builds a small convolutional neural network (CNN).
- Trains the model to classify digits from 0 to 9.
- Prints epoch loss, test accuracy, training time, and the detected device.
- Saves model weights to `checkpoints/cnn/mnist_cnn.pt`.

**What to learn**

- How convolution and pooling layers process image tensors.
- How a CNN produces class scores.
- How to evaluate classification accuracy on a separate test dataset.
- How the same FSDP training pattern can be applied to an image model.

**Important note:** MNIST is a standard educational dataset. This small CNN is a learning example, not a production computer-vision system.

### 3. `03_lstm_fsdp.py` — Character-level language model

**What it does**

- Builds a character vocabulary from a small, repeated sample text included in the script.
- Converts characters to integer IDs.
- Trains an embedding + LSTM + linear-layer model to predict the next character.
- Prints training loss, validation loss, perplexity, training time, and the detected device.
- Generates a short character sequence.
- Saves model weights and the vocabulary to `checkpoints/lstm/lstm_lm.pt`.

**What to learn**

- Character tokenization and sequence windows.
- How an LSTM uses earlier characters to predict the next character.
- Cross-entropy loss and perplexity.
- Basic autoregressive text generation.

**Important note:** This is a tiny character-level demonstration, not a GPT model, not a foundation model, and not a general-purpose language model. Its output may be repetitive or meaningless because the training corpus is very small.

## Requirements

- Linux environment (the examples are intended for an AWS EC2 Linux instance).
- Python version supported by the installed PyTorch release.
- An NVIDIA GPU and a compatible NVIDIA driver for the CUDA training path used by these scripts.
- A CUDA-enabled PyTorch build.
- Internet access for downloading MNIST the first time the CNN script runs.

Check the exact PyTorch, Python, and CUDA installation compatibility before installing. Official guide: <https://pytorch.org/get-started/locally/>.

## Setup on an EC2 Linux instance

### 1. Check the machine and GPU

```bash
cat /etc/os-release
python3 --version
nvidia-smi
```

If `nvidia-smi` fails, fix the instance/driver setup before running the training scripts.

### 2. Create a virtual environment

```bash
mkdir -p ~/fsdp-lab
cd ~/fsdp-lab

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch and dependencies

First install the CUDA-enabled PyTorch build that matches your system using the official PyTorch selector: <https://pytorch.org/get-started/locally/>.

Then install the additional packages:

```bash
python -m pip install numpy scikit-learn torchvision
```

Make sure the installed `torchvision` version is compatible with the installed PyTorch version.

### 4. Verify PyTorch GPU access and FSDP import

```bash
python - <<'PY'
import torch
from torch.distributed.fsdp import FullyShardedDataParallel

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("FSDP import: OK")
PY
```

Do not start training until `CUDA available: True` is shown for the intended GPU setup.

### 5. Create local output directories

```bash
mkdir -p data logs
mkdir -p checkpoints/ml checkpoints/cnn checkpoints/lstm
```

## Run the examples

Run commands from the repository root with the virtual environment activated. Use the exact filename shown in each command.

### Tabular MLP

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 ./01_ml_fsdp.py
```

### CNN on MNIST

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 ./02_cnn_fsdp.py
```

### Character-level LSTM

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 ./03_lstm_fsdp.py
```

`--nproc_per_node=1` starts one training process. These commands are intended for a single-node, single-GPU learning run. Actual results and run times depend on the machine and installed software.

To monitor GPU usage from another terminal:

```bash
watch -n 1 nvidia-smi
```

## Outputs

Expected checkpoint paths:

```text
checkpoints/
├── ml/model.pt
├── cnn/mnist_cnn.pt
└── lstm/lstm_lm.pt
```

The scripts print metrics to the terminal. The precise accuracy, losses, generated text, and execution time are not fixed and should not be copied into documentation as guaranteed results.

The current examples save model weights (and the LSTM vocabulary). They do not yet save all optimizer state and training progress needed for a fully resumable training checkpoint.

## What single-GPU FSDP does—and does not—show

FSDP is designed to manage/shard model states across distributed processes. In multi-GPU training, it can shard parameters, gradients, and optimizer states, using distributed communication as needed.

These examples use one process on one GPU to practise FSDP wrapping and the surrounding training workflow. They **do not demonstrate multi-GPU scaling**, and a single-GPU run should not be presented as proof of multi-GPU memory savings or performance improvements.

For deeper study, see the official PyTorch FSDP documentation: <https://pytorch.org/docs/stable/fsdp.html>.

## Troubleshooting

### `can't open file ... No such file or directory`

Check your current directory and the exact filename:

```bash
pwd
ls -la
```

Run the file that actually exists. For example, the CNN script is `02_cnn_fsdp.py`, not `01_cnn_fsdp.py`.

### `CUDA available: False`

Check `nvidia-smi`, then verify that the installed PyTorch build supports the CUDA configuration on your system. Reinstall PyTorch using the official selector if needed.

### MNIST download fails

The first CNN run needs internet access to download the dataset. Check network access and available disk space, then run it again.

### A Python or FSDP API error appears

Record the complete traceback and these version details:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip show torch torchvision
```

Check the official PyTorch documentation for the API that matches your installed version.

## Current scope and future improvements

This repository is intentionally small and educational. Possible future exercises include:

- Add reproducible seeds and configuration options.
- Save optimizer state and support resuming a run.
- Add structured logging and experiment tracking.
- Add mixed precision and FSDP wrapping policies.
- Compare single-GPU and multi-GPU runs on suitable hardware.
- Measure throughput and GPU memory usage systematically.

## License

No license is specified yet. Add a license file before allowing others to reuse, modify, or redistribute the code. Do not imply that the repository is open-source under a particular license until one has been chosen.

---

**Purpose:** Learn PyTorch FSDP concepts through small, inspectable examples—not to provide a trained production model.

