import torch
import torch.nn as nn
import copy
from utils.log import logger


def train_one_epoch(model, loader, optimizer, loss_fn, max_grad_norm=1.0):
    """One pass over the training data. Returns the mean loss."""
    model.train()
    total = 0.0
    for x, timespans, y in loader:
        loss = loss_fn(model(x, timespans)[:, -1, 0], y)   #prediction for each window's last day
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    """Predicted and actual next day returns for every window."""
    model.eval()
    preds, targets = [], []
    for x, timespans, y in loader:
        preds.append(model(x, timespans)[:, -1, 0])
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def fit(model, train_loader, val_loader, optimizer, loss_fn, epochs=100, patience=10):
    """Train with early stopping, then restore the weights with the lowest validation loss."""
    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        pred, target = predict(model, val_loader)
        val_loss = loss_fn(torch.from_numpy(pred), torch.from_numpy(target)).item()

        if val_loss < best_loss:
            best_loss, best_state, best_epoch, waited = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1

        logger.info(f"epoch {epoch:3d} | train {train_loss:.4e} | val {val_loss:.4e}{'  New Best' if waited == 0 else ''}")
        if waited >= patience:
            break

    model.load_state_dict(best_state)
    logger.info(f"best model from epoch {best_epoch} | val {best_loss:.4e}")