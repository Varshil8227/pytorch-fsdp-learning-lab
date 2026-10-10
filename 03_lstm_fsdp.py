
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

TEXT = (
    "machine learning systems learn patterns from data. "
    "deep learning uses neural networks. "
    "language models predict the next token. "
    "pytorch helps engineers train models on GPUs. "
) * 100

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)

    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    seq_len = 40
    batch_size = 64

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(chars), 64)
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            )
            self.fc = nn.Linear(128, len(chars))

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            return self.fc(x)

    def get_batch(source):
        starts = torch.randint(
            0, len(source) - seq_len - 1, (batch_size,)
        )
        x = torch.stack([source[s:s + seq_len] for s in starts])
        y = torch.stack([
            source[s + 1:s + seq_len + 1] for s in starts
        ])
        return x.to(device), y.to(device)

    model = FSDP(LSTMModel().to(device), device_id=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        steps = 300

        for _ in range(steps):
            x, y = get_batch(train_data)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, len(chars)),
                y.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch}/3 | "
                  f"Train loss: {total_loss / steps:.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        x, y = get_batch(val_data)
        logits = model(x)
        val_loss = criterion(
            logits.reshape(-1, len(chars)),
            y.reshape(-1),
        ).item()

    perplexity = torch.exp(torch.tensor(val_loss)).item()

    if rank == 0:
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Training time: {elapsed:.2f} seconds")
        print(f"Device: {device} | {torch.cuda.get_device_name(device)}")

    # Generate characters autoregressively.
    model.eval()
    with torch.no_grad():
        context = torch.tensor(
            [[stoi["m"]]], dtype=torch.long, device=device
        )
        generated = context.clone()

        for _ in range(200):
            logits = model(context)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            context = torch.cat([context, next_id], dim=1)
            context = context[:, -seq_len:]

        text = "".join(itos[i] for i in generated[0].tolist())

    if rank == 0:
        print("\nGenerated text:")
        print(text)

    os.makedirs("checkpoints/lstm", exist_ok=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state = model.state_dict()

    if rank == 0:
        torch.save(
            {"model_state_dict": state, "vocabulary": chars},
            "checkpoints/lstm/lstm_lm.pt",
        )
        print("Saved: checkpoints/lstm/lstm_lm.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
