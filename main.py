import argparse
import importlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import get_data_loaders
from train import train_epoch, evaluate
from torch.optim.lr_scheduler import OneCycleLR
from torchsummary import summary

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensures reproducibility for cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_name: str, device: torch.device):
    if model_name == "model1":
        module = importlib.import_module("model1")
        return module.build_model(device)
    elif model_name == "model2":
        module = importlib.import_module("model2")
        return module.build_model(device)
    elif model_name == "model3":
        module = importlib.import_module("model3")
        return module.build_model(device)
    raise ValueError(f"Unknown model '{model_name}'.")


def main():
    parser = argparse.ArgumentParser(description="MNIST Training")
    parser.add_argument("--model", type=str, default="model1", help="Model to use: model1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    device = get_device(prefer_cuda=not args.no_cuda)

    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=2,
        pin_memory=True,
        shuffle_train=True,
        model_name=args.model,
    )

    model = build_model(args.model, device)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    summary(model, input_size=(1, 28, 28))
    
    if args.model == "model3":
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=0.1,
                            steps_per_epoch=len(train_loader),
                            epochs=args.epochs)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        tr_loss, tr_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, device, test_loader, criterion)
        if args.model == "model3":
            scheduler.step(te_loss)   # pass validation loss
        else:
            scheduler.step()


        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        test_losses.append(te_loss)
        test_acc.append(te_acc)

        print(
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}%"
        )


if __name__ == "__main__":
    main()
