import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.hybrid import LiquidMamba
from train import fit, predict
from utils.dataset import build_datasets
from utils.eval import evaluation_metric
from utils.log import log_scores, logger, setup

#Settings without argument from the commandline. No need at the moment.

#Name of the dataset to use without .csv extension inside data/
dataset = "BTC"

#Logging. Turn on and off console/file output. File logs directory is logs/ by default.
setup(console=True, file=True, name=dataset)

#Seed, for reproducible results.
seed = 0
torch.manual_seed(seed)

#Config stops here.

#Data
train, val, test, _ = build_datasets(dataset)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
val_loader = DataLoader(val, batch_size=64)
test_loader = DataLoader(test, batch_size=64)

#Model
model = LiquidMamba(in_dim=train.n_features)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
logger.info(f"{dataset}: window counts: train: {len(train)}, val {len(val)}, test {len(test)}")
logger.info(f"features {', '.join(train.feature_names)}, parameters {sum(p.numel() for p in model.parameters()):,}")
logger.info(f"seed {seed}")

#Training
fit(model, train_loader, val_loader, optimizer, nn.MSELoss())

#Test
y_hat, y_test = predict(model, test_loader)
logger.info("test")
log_scores(evaluation_metric(y_test, y_hat), evaluation_metric(y_test, np.zeros_like(y_test)))
