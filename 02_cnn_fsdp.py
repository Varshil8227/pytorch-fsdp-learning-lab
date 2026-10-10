
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=2
    )

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = FSDP(CNN().to(device), device_id=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        if rank == 0:
            print(f"Epoch {epoch}/3 | "
                  f"Loss: {total_loss / total:.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if rank == 0:
        print(f"Test accuracy: {100 * correct / total:.2f}%")
        print(f"Training time: {elapsed:.2f} seconds")
        print(f"Device: {device} | {torch.cuda.get_device_name(device)}")

    os.makedirs("checkpoints/cnn", exist_ok=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state = model.state_dict()

    if rank == 0:
        torch.save(state, "checkpoints/cnn/mnist_cnn.pt")
        print("Saved: checkpoints/cnn/mnist_cnn.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
