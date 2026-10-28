# Bench

## Results

All runs: 10 epochs, weighted clauses.
Predict: mean sec over all epochs. 
Acc/metric: only last epoch.

CPU Runs: Apple M3, 16 GB RAM
CUDA Runs: Nvidia H100, 80GB VRAM

### Non-coalesced
#### CPU

(YTM / TMU)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak RAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 1.33s / 2.21s | 0.78s / 1.92s | 0.05s / 0.58s | 479 / 740 | 92.35% / 92.12% |
| MNIST conv | 2.93s / 9.45s | 3.05s / 4.88s | 0.70s / 17.15s | 534 / 1509 | 96.50% / 96.56% |
| FMNIST conv | 16.09s / 83.06s | 17.08s / 72.09s | 2.05s / 64.95s | 808 / 3691 | 87.15% / 86.24% |
| IMDb | 223.24s / 86.91min* | 70.76s / - | 3.43s / 4.14s* | 2134 / 3680* | 86.24% / - |

\* Ran only for 1 epoch.

#### CUDA

(YTM / TMU)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak VRAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 2.87 ± 0.01s / 20.85s | 2.87 ± 0.01s / 16.25s | 0.01 ± 0.00s / 3.70s | 604 ± 1 / 4702 | 92.13 ± 0.23% / 92.13% |
| MNIST conv | 2.96 ± 0.02s / 61.68s | 2.96 ± 0.01s / 38.71s | 0.01 ± 0.00s / 19.87s | 628 ± 3 / 4702 | 96.47 ± 0.11% / 96.46% |
| FMNIST conv | 3.92 ± 0.01s / 107.32s | 3.53 ± 0.03s / 71.19s | 0.05 ± 0.00s / 37.48s | 672 ± 19 / 4702 | 86.96 ± 0.18% / 86.54% |
| IMDb | 9.00 ± 0.08s / 108.71s | 3.15 ± 0.01s / 49.12s | 0.19 ± 0.00s / 20.39s | 1447 ± 24 / 4702 | 85.94 ± 0.28% / 86.18% |
| CIFAR-10 conv | 4.39 ± 0.07s / 147.47s | 4.29 ± 0.15s / 106.47s | 0.33 ± 0.00s / 43.88s | 969 ± 33 / 4702 | 59.04 ± 0.27% / 59.72% |
| AG News | 6.90 ± 0.06s / 83.92s | 6.40 ± 0.01s / 66.55s | 0.02 ± 0.00s / 2.32s | 859 ± 16 / 4702 | 89.83 ± 0.17% / 89.36% |


### Coalesced
#### CPU

(YTM / TMU)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak RAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 10.73s / 7.87s | 4.72s / 5.42s | 0.06s / 0.21s | 483 / 765 | 94.80% / 95.41% |
| MNIST conv | 13.93s / 28.04s | 13.42s / 19.55s | 0.61s / 5.77s | 533 / 1531 | 97.61% / 97.77% |
| FMNIST conv | 28.25s / 448.25s | 23.84s / 353.52s | 1.31s / 82.87s | 805 / 3231 | 87.78% / 87.65% |


#### CUDA

(YTM / PySparseCoalescedTsetlinMachineCUDA)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak VRAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 3.21 ± 0.25s / 25.40s | 3.30 ± 0.22s / 14.44s | 0.01 ± 0.00s / 1.77s | 605 ± 1 / 610 | 95.19 ± 0.11% / 95.06% |
| MNIST conv | 2.99 ± 0.03s / 31.06s | 2.71 ± 0.01s / 25.77s | 0.01 ± 0.00s / 2.21s | 628 ± 3 / 715 | 97.74 ± 0.12% / 97.74% |
| FMNIST conv | 4.76 ± 0.11s / 99.74s | 4.22 ± 0.25s / 96.56s | 0.04 ± 0.00s / 12.48s | 672 ± 19 / 1495 | 87.71 ± 0.10% / 87.52% |
| CIFAR-10 conv | 5.03 ± 0.03s / 98.11s | 4.74 ± 0.02s / 94.88s | 0.33 ± 0.01s / 14.43s | 969 ± 33 / 1632 | 62.04 ± 0.29% / 62.29% |
| CelebA conv | 96.39 ± 21.02s / >3.5hr* | 92.55 ± 9.03s / - | 4.10 ± 1.06s / - | 4295 ± 208 / - | 56.02 ± 0.54% / - |
| AG News | 11.37 ± 0.04s / 68.26s | 9.68 ± 0.01s / 51.38s | 0.01 ± 0.00s / 0.73s | 860 ± 16 / 1116 | 89.52 ± 0.19% / 89.41% |
| GoEmotions | 61.79 ± 0.13s / 435.38s | 50.71 ± 0.08s / 346.05s | 0.06 ± 0.00s / 2.06s | 1679 ± 101 / 8187 | 35.00 ± 0.70% / 35.47% |

\* Epoch 1 still running.

Note:
- CPU threads: YTM CPU uses 1 thread for MNIST and 8 threads (`cpu:8`) for FMNIST and IMDb.
- CIFAR-10, AG News, CelebA and GoEmotions are CUDA only.
- CelebA and GoEmotions are multi-label and coalesced only. Acc is that case is macro F1.
- PySparse has no seed argument, so results for it can vary.
- YTM CUDA rows: mean ± std over 5 seeds (10, 20, 30, 40, 50). Other cells: single run, seed 10.

## Hyperparameters

| Experiment | Classes | Input | Patch | Coalesced clauses / T | Non-coalesced clauses per class / T | s | q |
|---|---|---|---|---|---|---|---|
| MNIST flat | 10 | >75 binary | - | 500 / 1000 | 50 / 100 | 10 | |
| MNIST conv | 10 | >75 binary | 10x10 | 500 / 1000 | 50 / 100 | 10 | |
| FMNIST conv | 10 | 8-level thermometer | 3x3 | 6000 / 10000 | 600 / 1000 | 10 | |
| CIFAR-10 conv | 10 | adaptive threshold RGB | 5x5 | 15000 / 20000 | 1500 / 2000 | 10 | |
| IMDb | 2 | 12000 chi2 1-2 grams | - | - | 5000 / 10000 | 1.1 | |
| AG News | 4 | 2000 chi2 lemmatized | - | 8000 / 40000 | 2000 / 10000 | 2.0 | |
| CelebA conv | 40 | 64x64 RGB 8-level thermometer | 3x3 | 80000 / 80000 | - | 30 | 10 |
| GoEmotions | 28 | 10000 1-2 grams | - | 22400 / 25000 | - | 2.0 | 1.5 |


## Commands

From the repo root. CPU runs use the `bench` environment, CUDA runs use `bench-cuda`.

- MNIST flat

  ```bash
  pixi run -e bench python -m bench.ytm.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench python -m bench.ytm.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CUDA
  ```

- MNIST conv

  ```bash
  pixi run -e bench python -m bench.ytm.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench python -m bench.ytm.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CUDA
  ```

- FMNIST conv

  ```bash
  pixi run -e bench python -m bench.ytm.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --device cpu:8
  pixi run -e bench python -m bench.tmu.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench python -m bench.ytm.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --device cpu:8
  pixi run -e bench python -m bench.tmu.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CUDA
  ```

- IMDb

  ```bash
  pixi run -e bench python -m bench.ytm.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 10 --seed 10 --device cpu:8
  pixi run -e bench python -m bench.tmu.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 1 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 10 --seed 10 --platform CUDA
  ```

- CIFAR-10 conv

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.cifar10 --patch 5 --coalesced 1 --n_clauses 15000 --T 20000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.cifar10 --patch 5 --coalesced 1 --n_clauses 15000 --T 20000 --s 10 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.cifar10 --patch 5 --coalesced 0 --n_clauses 1500 --T 2000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.cifar10 --patch 5 --coalesced 0 --n_clauses 1500 --T 2000 --s 10 --epochs 10 --seed 10 --platform CUDA
  ```

- AG News

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.ag_news --max_features 2000 --coalesced 1 --n_clauses 8000 --T 40000 --s 2 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.ag_news --max_features 2000 --coalesced 1 --n_clauses 8000 --T 40000 --s 2 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.ag_news --max_features 2000 --coalesced 0 --n_clauses 2000 --T 10000 --s 2 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.ag_news --max_features 2000 --coalesced 0 --n_clauses 2000 --T 10000 --s 2 --epochs 10 --seed 10 --platform CUDA
  ```

- CelebA conv

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.celeba --patch 3 --levels 8 --n_clauses 80000 --T 80000 --s 30 --q 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.celeba --patch 3 --levels 8 --n_clauses 80000 --T 80000 --s 30 --q 10 --epochs 10
  ```

- GoEmotions

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.go_emotions --max_features 10000 --n_clauses 22400 --T 25000 --s 2 --q 1.5 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.go_emotions --max_features 10000 --n_clauses 22400 --T 25000 --s 2 --q 1.5 --epochs 10
  ```
