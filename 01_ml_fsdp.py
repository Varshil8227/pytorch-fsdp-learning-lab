
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)
    np.random.seed(42)

    X, y = make_classification(
        n_samples=12000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test.astype(np.int64)),
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

    model = FSDP(MLP().to(device), device_id=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

        if rank == 0:
            print(f"Epoch {epoch}/5 | "
                  f"Loss: {total_loss / total_samples:.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model.eval()
    correct = 0
    count = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            predictions = model(xb).argmax(dim=1)
            correct += (predictions == yb).sum().item()
            count += yb.size(0)

    if rank == 0:
        accuracy = 100 * correct / count
        print(f"Test accuracy: {accuracy:.2f}%")
        print(f"Training time: {elapsed:.2f} seconds")
        print(f"Device: {device} | {torch.cuda.get_device_name(device)}")

    os.makedirs("checkpoints/ml", exist_ok=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state = model.state_dict()

    if rank == 0:
        torch.save(state, "checkpoints/ml/model.pt")
        print("Saved: checkpoints/ml/model.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
