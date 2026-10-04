import logging
from datetime import datetime
from pathlib import Path
from utils.eval import METRIC_NAMES

logger = logging.getLogger("liquidmamba")


def setup(console, file, name, log_directory="logs"):
    """
    Configures the project logger. Console and file output can both be switched on or off.
    The log file is written to <log_directory>/<data_name>_<time>.log.
    May add wandb connection as the original project did later on.
    """

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    logger.disabled = not (console or file)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handlers = []
    if console:
        handlers.append(logging.StreamHandler())
    if file:
        Path(log_directory).mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_directory) / f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"))

    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def log_scores(model_scores, zero_scores):
    """Log evaluation_metric results for the model next to the zero forecast."""
    for name, model, zero in zip(METRIC_NAMES, model_scores, zero_scores):
        logger.info(f"{name:5s} model {model:.4e} | zero {zero:.4e}")