import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.hybrid import LiquidMamba
from train import fit, predict
from utils.dataset import build_datasets
from utils.eval import evaluation_metric
from utils.log import log_scores, logger, setup


def direction_baseline(train, y):
    """Directional accuracy of always predicting the most common direction in the training targets."""
    train_targets = train.targets.numpy()[train.ends]
    majority = 1 if np.mean(train_targets > 0) >= 0.5 else -1
    moved = y != 0
    return np.mean(np.sign(y[moved]) == majority)


def run(dataset, seed):
    """Train and evaluate LiquidMamba on one dataset with one seed. Returns validation and test results."""

    torch.manual_seed(seed)

    #Data
    train, val, test, _ = build_datasets(dataset)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64)
    test_loader = DataLoader(test, batch_size=64)

    #Model
    model = LiquidMamba(in_dim=train.n_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    logger.info(f"{dataset}, seed {seed}")
    logger.info(f"window counts: train {len(train)}, val {len(val)}, test {len(test)}")
    logger.info(f"features {', '.join(train.feature_names)}, parameters {sum(p.numel() for p in model.parameters()):,}")

    #Zero forecast MSE, the reference for every epoch's train and val loss.
    train_zero = np.mean(train.targets.numpy()[train.ends] ** 2)
    val_zero = np.mean(val.targets.numpy()[val.ends] ** 2)
    logger.info(f"zero forecast MSE: train {train_zero:.4e} | val {val_zero:.4e}")

    #Training
    fit(model, train_loader, val_loader, optimizer, nn.MSELoss())

    #Validation and test, both with the best model.
    results = {}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        y_hat, y = predict(model, loader)
        scores = evaluation_metric(y, y_hat)
        zero_scores = evaluation_metric(y, np.zeros_like(y))
        dir_baseline = direction_baseline(train, y)

        logger.info(split)
        log_scores(scores, zero_scores)
        logger.info(f"majority direction baseline {dir_baseline:.4e}")

        results[split] = {"scores": scores, "dir_baseline": dir_baseline,
                          "pred_std": float(np.std(y_hat)), "pred_up": float(np.mean(y_hat > 0))}

    return results


if __name__ == "__main__":
 
    #Settings for a single run, used with python main.py and no arguments.

    #Name of the dataset to use without .csv extension inside data/
    dataset = "EURTRY_ret"
    #Seed, for reproducible results.
    seed = 0
 
    #Logging. Turn on and off console/file output. File logs directory is logs/ by default.
    setup(console=True, file=True, name=dataset)
 
    run(dataset, seed)
