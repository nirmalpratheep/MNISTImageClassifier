import torch
import torch.nn as nn
from tqdm import tqdm


def get_correct_pred_count(predictions: torch.Tensor, labels: torch.Tensor) -> int:
    return predictions.argmax(dim=1).eq(labels).sum().item()


def train_epoch(model, device, train_loader, optimizer, criterion, progress: bool = True):
    model.train()
    data_iter = tqdm(train_loader) if progress else train_loader

    epoch_loss = 0.0
    correct = 0
    processed = 0

    for batch_index, (data, target) in enumerate(data_iter):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        predictions = model(data)
        loss = criterion(predictions, target)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        correct += get_correct_pred_count(predictions, target)
        processed += len(data)

        if progress:
            data_iter.set_description(
                desc=f'Train: Loss={loss.item():0.4f} Batch={batch_index} Accuracy={100 * correct / processed:0.2f}'
            )

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100.0 * correct / processed
    return avg_loss, accuracy


def evaluate(model, device, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            total_loss += criterion(outputs, target).item()
            correct += get_correct_pred_count(outputs, target)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    return avg_loss, accuracy
