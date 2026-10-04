from datetime import datetime
from pathlib import Path

import pandas as pd

from main import run
from utils.eval import METRIC_NAMES
from utils.log import logger, setup

if __name__ == "__main__":

    #Every dataset in data/ named <ASSET>_<type>.csv
    datasets = sorted(p.stem for p in Path("data").glob("*_*.csv"))
    seeds = [0, 1, 2]

    setup(console=True, file=True, name="batch")
    logger.info(f"datasets {', '.join(datasets)} | seeds {seeds}")

    Path("results").mkdir(exist_ok=True)
    results_file = f"results/batch_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"

    rows = []
    for dataset in datasets:
        for seed in seeds:

            try:
                results = run(dataset, seed)
            except Exception:
                logger.exception(f"{dataset}, seed {seed} failed, continuing with the next run")
                continue

            #One row per run: val_MSE, ..., test_DIR, plus the direction baselines.
            row = {"dataset": dataset, "seed": seed}
            for split, result in results.items():
                row.update({f"{split}_{name}": value for name, value in zip(METRIC_NAMES, result["scores"])})
                row[f"{split}_DIR_baseline"] = result["dir_baseline"]
                row[f"{split}_pred_std"] = result["pred_std"]
                row[f"{split}_pred_up"] = result["pred_up"]
            rows.append(row)

            #Rewritten after every run, so finished runs are kept if the batch stops early.
            pd.DataFrame(rows).to_csv(results_file, index=False)

    #Mean and standard deviation across seeds, per dataset. Validation first, since decisions are made on it.
    columns = ["val_R2", "val_MSE", "val_DIR", "val_DIR_baseline",
               "test_R2", "test_MSE", "test_DIR", "test_DIR_baseline"]
    summary = pd.DataFrame(rows).groupby("dataset")[columns].agg(["mean", "std"])
    logger.info("summary across seeds\n" + summary.to_string(float_format=lambda x: f"{x:.4f}"))
    logger.info(f"results saved to {results_file}")