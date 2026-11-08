import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.liquidmamba import LiquidMamba
from train import fit, predict
from utils.dataset import build_datasets
from utils.eval import evaluate
from utils.log import logger, setup


def run(dataset, seed, lwindow, swindow):
    """Train and evaluate LiquidMamba on one dataset with one seed. Returns validation and test results."""

    torch.manual_seed(seed)

    #Data
    train, val, test, _ = build_datasets(dataset, window=lwindow)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64)
    test_loader = DataLoader(test, batch_size=64)

    #Model
    model = LiquidMamba(input_size=train.n_features, short_window=swindow)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    logger.info(f"{dataset}, seed {seed}")
    logger.info(f"window counts: train {len(train)}, val {len(val)}, test {len(test)}")
    logger.info(f"features {', '.join(train.feature_names)}, parameters {sum(p.numel() for p in model.parameters()):,}")

    #Zero forecast MSE, the reference for every epoch's train and val loss.
    train_zero = np.mean(train.targets.numpy()[train.ends] ** 2)
    val_zero = np.mean(val.targets.numpy()[val.ends] ** 2)
    logger.info(f"zero forecast MSE: train {train_zero:.8f} | val {val_zero:.8f}")

    #Training
    fit(model, train_loader, val_loader, optimizer, nn.MSELoss())

    #Validation and test, both with the best model.
    #Baseline always predicts 0 return and training majority direction.
    train_targets = train.targets.numpy()[train.ends]
    majority = 1.0 if np.mean(train_targets > 0) >= 0.5 else -1.0

    results = {}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        y_hat, y = predict(model, loader)
        results[split] = {"model": evaluate(y, y_hat),
                          "zero": evaluate(y, np.zeros_like(y)),
                          "majority": evaluate(y, np.full_like(y, majority * 1e-4))}

        logger.info(split)
        for metric in results[split]["model"]:
            row = " | ".join(f"{name} {results[split][name][metric]:.6f}" for name in ("model", "zero", "majority"))
            logger.info(f"{metric:8s} {row}")

    return results


if __name__ == "__main__":
 
    #Name of the dataset to use without .csv extension inside data/
    dataset = "SPY_ti"
    #Seed, for reproducible results.
    seed = 0
    #Long window for mamba. 
    lwindow = 63
    #Short window for CfC.
    swindow = 5
    #Logging. Turn on and off console/file output. File logs directory is logs/ by default.
    setup(console=True, file=True, name=dataset)
 
    run(dataset, 0, lwindow, swindow)
