# LiquidMamba

Next-day return forecasting for financial time series with a hybrid of CfC (Closed-form Continuous-time Neural Networks) and Mamba (Linear-Time Sequence Modeling with Selective State Spaces).

The project asks whether combining a continuous time recurrent network which can account for irregular time gaps between observations and Mamba's selective memory can forecast next day returns better than baselines, across assets from different classes.

---

## Credits and branches

This repository is a fork of Abdullah Enes Doruk and Ahmet Mete Dokgöz's original [LiquidMamba project](https://github.com/enesdoruk/LiquidMamba)

- main branch contains the original code unchanged, with only a `requirements.txt` added along with `__pycache__` folders added to the .gitignore file for convenience.
- dev branch contains the work described in this README.

---

## Repository structure

```
LiquidMamba/
|--- main.py               one run: run(dataset, seed), plus single-run settings
|--- run_batch.py          many runs: every dataset × several seeds, results CSV and summary
|--- train.py              training loop, prediction, early stopping
|--- model/
│   |--- liquidnet.py      LiquidNet: wrapper around the ncps CfC network
│   |--- hybrid.py         LiquidMamba: the hybrid model
|--- utils/
│   |--- dataset.py        windowed dataset, chronological splits, standardization
│   |--- eval.py           evaluation metrics
│   |--- log.py            console and file logging
|--- data/                 prepared datasets, <ASSET>_<type>.csv |
|--- logs/                 one log file per run or batch         |--- untracked with .gitignore. Make sure to create the folders.
|--- results/              one results CSV per batch             |
|--- dataset_convention    rules every dataset must follow
|--- requirements.txt      pinned dependencies
```

## Setup

```
pip install -r requirements.txt
```

## Running

From the project root:

```
python main.py        # for one dataset and one seed 
python run_batch.py   # for every dataset in data/ with multiple seeds
```

A single run takes about 2 to 5 minutes on CPU, depending on the dataset.

---

## Data

### Sources and preparation

Data is downloaded and turned into features in  [ltsf](https://github.com/er2rk/ltsf)

1. Download: daily OHLCV data from Yahoo Finance via `yfinance`, with prices adjusted for splits and dividends..
2. Features: two datasets are built from the same raw data in one pass, and cleaned together so they always cover exactly the same dates:
   - Returns (`<ASSET>_ret.csv`): open, high, low and close relative to the previous close, and the change in log volume.
   - Technical indicators (`<ASSET>_ti.csv`): one-day return, MACD, Aroon oscillator, RSI, Bollinger band width, and the A/D oscillator. All are computed with TA-Lib.

Volume-based features are left out for assets without meaningful volume (eg. forex, brent).

### Dataset convention

Every dataset should follow the rules in `dataset_convention`:

1. Datasets should be cleaned for missing values before they are fed into the program.
2. Datasets do not need to be standardized or normalized. Scaling happens in utils/dataset.build_datasets.
3. Datasets should contain a "date" column in the format (YYYY-MM-DD). This might be generalized later for order book data.
4. Features at row t should onlt contain information available at the end of day t.
5. Datasets should specifically contain a day t's closing price at row t, as we are going to be computing our label from it.
6. The label 'target' at day t is the return from day t to day t+1. Calculated with: y_t = (close_t+1 / close_t) - 1. 

Every column except `date` and `close` is used as a feature. The raw close is only used to compute the label.
