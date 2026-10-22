# Reproducing results

Run commands from the repository root. [data/](data/README.md) holds the saved per-seed and per-repetition results, grouped by paper section. Each section folder provides summary CSVs and links to individual test records. Write new experiments to `results/` to keep the supplied data unchanged.

## Saved results: no training or GPU required

| Paper result | Command |
| --- | --- |
| All prepared paper summaries | `python results.py summaries --data data --out results/paper_tables` |
| Tables 1–3, plus all 784 onset cells | `python results.py onsets --data data --out results/onsets.csv` |
| Table 5 and Appendix C controls | `python results.py type-i --data data --out results/type_i.csv` |
| Table 6, Figure 3(a), Appendix D | `python results.py synthetic --data data` |
| Table 7, small sample size | `python results.py small-sample --data data --out results/small_sample.csv` |
| Table 8, reversed reference | `python results.py reversed --data data --out results/reversed.csv` |
| Appendix I, MMD on VRT scores | `python results.py score-transform --data data --out results/score_transform.csv` |
| Table 9, CPU runtime | `python results.py runtime --data data --out results/runtime.csv` |
| Result schemas and manifest hashes | `python results.py validate --data data` |

The onset CSV also includes Table 4's Schilling and Henze comparisons (`stat=schilling` or `henze`). It records evaluated checkpoints, incomplete seed sets and missing checkpoints around or after detection. The reference CSV is `docs/onsets_final.csv`.

Each checkpoint uses analysis seeds 42–91. VRT, MMD, ED, Schilling and Henze reject when their median p-value is below 0.05. C2ST additionally requires median held-out accuracy above 0.55. MMDAgg requires at least 60% of seeds to reject. Sustained onset is the first evaluated checkpoint that rejects and remains rejecting at every later evaluated checkpoint. Rounded reversed-VRT and Henze p-values are recovered on the recorded permutation lattice before aggregation.

The Type-I comparison uses single-test decisions, including C2ST p-values without the accuracy threshold. Its random-partition and original-split results are reported separately. The separate 50-seed control reports decisions across randomly initialized networks.

## Installation

Use Python 3.11. The direct VRT Python example and saved-result summaries need only NumPy and SciPy, as shown in the [quick start](README.md#reproduce-saved-tables). All `evaluate.py` commands, including VRT-only evaluation, need the full dependencies below. Install these also for feature extraction, training, controls, synthetic experiments and tests:

```bash
python -m pip install -r requirements.txt
```

The experiment environment is recorded in [ENVIRONMENT.md](ENVIRONMENT.md).

## Direct Python use

Supply two arrays of shape `(N, D)`, with one embedding per row and the same feature dimension `D`. For the main protocol, use at least 10,000 training rows and exactly 10,000 test rows. Each row must be finite and nonzero.

```python
import numpy as np
from common import l2n, validate_features
from vrt import vrt_full

train = np.load("train.npy").astype(np.float64)
test = np.load("test.npy").astype(np.float64)
validate_features(train, test)

statistic, p_value = vrt_full(
    l2n(train), l2n(test), D=train.shape[1], seed=42
)
print(statistic, p_value)
```

This uses 2,000 training reference points, 8,000 disjoint training queries, 10,000 test queries, 60 neighbours and 2,000 permutations. The statistic is the test-minus-training median score difference; the p-value is two-sided. For 1,500 points per split, use `vrt_low` with its default 500 reference points and 15 neighbours.

## New tests on feature arrays

An NPZ file must contain training and test arrays with the same feature dimension. Supported key pairs are `h_tr`/`h_te`, `X_tr`/`X_te` and `tr`/`te`. Main VRT needs at least 10,000 training rows and exactly 10,000 test rows. The small protocol needs at least 1,500 rows from each split.

Save the same `train.npy` and `test.npy` arrays in one NPZ file:

```python
import numpy as np

np.savez("features.npz", h_tr=np.load("train.npy"), h_te=np.load("test.npy"))
```

Run one VRT test; the runner normalizes the rows automatically:

```bash
python evaluate.py test \
  --features features.npz --model my_model --epochs 0 \
  --methods vrt --budget 10000 --seeds 42 --permutations 2000 \
  --out results/my_vrt.csv
```

This writes one row to `results/my_vrt.csv`, including `statistic` and the two-sided p-value `p`. Here `0` labels the supplied checkpoint; replace it with its actual epoch if known. `--model` is an output label; use distinct labels for different training runs or extraction modes.

The analysis seed controls reference/query sampling and permutations, not model training. A single test does not establish sustained onset; the paper evaluates 50 analysis seeds at each checkpoint using the decision rule above.

### Checkpoint trajectories

```bash
python evaluate.py test \
  --features 'features/r50_eval/ep{ep:03d}.npz' --model r50_eval \
  --epochs 0-200 --methods vrt,mmd,ed,c2st,mmdagg,knn,reversed,mmd-scores \
  --budget 10000 --seeds 42-91 --permutations 2000 \
  --protocol supervised --backend numpy --out results/r50_eval.csv
```

`knn` produces both Schilling and Henze results. At the main budget, MMDAgg uses 5,000 points per split. Use `--budget 1500` for the small-budget comparison, omitting `mmd-scores`, whose paper protocol uses the main budget. Use `--protocol simclr` for the SimCLR preprocessing choices.

For Table 3's unnormalized comparison, select MMD and ED with `--raw`:

```bash
python evaluate.py test \
  --features 'features/r50_eval/ep{ep:03d}.npz' --model r50_eval \
  --epochs 0-200 --methods mmd,ed --raw \
  --out results/r50_raw.csv
```

`--backend torch` uses CUDA when available for compatible baseline operations. VRT, C2ST and MMDAgg remain CPU computations. Output files are created exclusively; use a new output name for each run.

Aggregate new experiments with:

```bash
python results.py new-runs \
  --input results/r50_eval.csv results/r50_raw.csv \
  --seeds 42-91 --out results/new_onsets.csv
```

This format records method, budget, normalization, protocol, analysis seed and requested permutation/backend options. C2ST and Schilling do not use permutations; MMDAgg uses its own defaults. Separate settings are not pooled. When aggregating a smaller seed set, pass the same `--seeds` as the test run. Missing-seed checks are explicit; checkpoint coverage is measured against epochs present for that model in the supplied files. Checkpoints absent from every input cannot be inferred.

## Model and checkpoint sources

The models below are used in Section 4.1 and Tables 1–3. Download external weights from their authors; the models trained by us have [training commands below](#training-recipes).

| Model in the paper | Source and selection | `models.py --arch` |
| --- | --- | --- |
| ResNet20-FRN-SiLU / CIFAR-10 (`v7`) | [TorchUncertainty weights](https://huggingface.co/datasets/torch-uncertainty/Checkpoints/tree/main/models/cifar10-resnet20-frn-silu); version 7 | `frn_c10` |
| ResNet-18 / CIFAR-10 and CIFAR-100 zoos | [Full zoos and loading instructions](https://github.com/ModelZoos/ModelZooDataset#model-zoos); Zenodo snapshots: [CIFAR-10](https://zenodo.org/records/6974029), [CIFAR-100](https://zenodo.org/records/6977382). We use five runs per dataset from the `kaiming_uniform` zoos. | `r18_c10`, `r18_c100` |
| ResNet-50 / CIFAR-100 (`r50`) | Trained by us: `train.py r50` | `r50_c100` |
| Phase Zoo ResNet-50 / CIFAR-100 | [Checkpoint archive](https://drive.proton.me/urls/V2E66KY0JM#Pq5M06URN4EN), [author instructions](https://github.com/ModelZoos/PhaseTransitionModelZoo#dataset-access). Use `cifar_100-ResNet50_subset`, width 64, seed 0, batch sizes 8/128/1024, epochs 100/125/150. | `r50_c100` |
| SimCLR ResNet-50 / CIFAR-10 and CIFAR-100 | Trained by us using [SupContrast](https://github.com/HobbitLong/SupContrast) in SimCLR mode: `train.py simclr` | `simclr_c10`, `simclr_c100` |
| 5-layer CNN / CIFAR-10 (`wp`) | [CNN Wild Park](https://zenodo.org/records/12797219); trial containing `2f9ed_00036`. Keep its `params.json` and pass `--wp-config`. | `wp` |
| DeepResNet50 / ImageNet, simple and advanced augmentation | [Encoder weights](https://zenodo.org/records/13971513); the two ResNet50 pretraining schemes in `encoder_weights.zip` | `imagenet` |
| ResNet-18 / CIFAR-10, configurations A–D | Trained by us: `train.py regularization` | `r18_c10` |

The CIFAR-10 Zenodo snapshot contains epochs 1, 10 and 50; the CIFAR-100 snapshot contains only epoch 60. Use the full-zoo links for training trajectories. The labels `m0`–`m4` in our result files identify our selected runs, not upstream run IDs. The Hugging Face page supplies model weights; the full `v7` epoch series is not bundled here.

For the Phase Zoo, the selected trial names contain `bf0b6_00173` (batch 8), `bf0b6_00077` (batch 128) and `bf0b6_00005` (batch 1024); all use learning rate 0.1 and weight decay 0.0005.

Pass the downloaded file path to `--checkpoint-format`, with one `--epochs` value for a single checkpoint. For trajectories, use a pattern such as `checkpoints/<trial>/checkpoint_{ep:06d}/checkpoints` for the CIFAR and Phase zoos. Wild Park uses `checkpoint_{ep:06d}/checkpoint.pt`. These are Python format strings, not wildcard patterns. Keep each run in a separate directory and extract it separately.

## Feature extraction and BatchNorm comparisons

The extraction runner needs image data and compatible model checkpoints. CIFAR datasets must already be available in the supplied data directory. ImageNet uses `train/` and `val/` subdirectories. Features are extracted without stochastic image augmentation and saved before the SimCLR projection head.

```bash
python models.py --arch r50_c100 --data-root datasets \
  --checkpoint-format 'checkpoints/r50/checkpoint_{ep:03d}.pt' \
  --epochs 0-200 --bn-mode eval --out features/r50_eval
python models.py --arch r50_c100 --data-root datasets \
  --checkpoint-format 'checkpoints/r50/checkpoint_{ep:03d}.pt' \
  --epochs 0-200 --bn-mode train --out features/r50_train
```

Run `evaluate.py test` on each feature directory using different model labels and output files. The checkpoint weights are the same; the extraction mode changes. `python models.py --help` lists architectures, the random-initialization option and the ImageNet disjoint-training split.

## Training recipes

These commands provide the ResNet-50, regularization and SimCLR recipes. Training data and resulting checkpoints are separate from the released numerical results.

```bash
python train.py r50 --data-root datasets --out checkpoints/r50 --seed 42
python train.py regularization --data-root datasets \
  --out checkpoints/regularization --seed 42
python train.py simclr --data-root datasets --dataset cifar10 \
  --out checkpoints/simclr --seed 0 --batch_size 512 --learning_rate 0.25
```

The regularization recipe runs configurations A–D; `--configs A_baseline` selects one. SimCLR accepts `--dataset cifar100`; the explicit batch size and learning rate above match the paper. Run `python train.py simclr --help` for its training settings. SimCLR saves every epoch in a configuration-named subdirectory of `--out`.

For the SimCLR loss trajectories in Appendix F:

```bash
SIMCLR_CHECKPOINTS=checkpoints/simclr/SimCLR_cifar10_resnet50_lr_0.25_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm
python train.py evaluate-loss --dataset cifar10 --data-root datasets \
  --checkpoint-format "$SIMCLR_CHECKPOINTS/ckpt_epoch_{ep}.pth" \
  --epochs 1-1000 --out results/simclr_cifar10_loss.csv
```

Point the checkpoint pattern to the directory produced by training. This evaluates two fixed augmented views of 10,000 training and 10,000 test images, using complete batches of 512 images in both BatchNorm modes. The released curves are in [data/08_per_model_trajectories/records/simclr_loss/](data/08_per_model_trajectories/records/simclr_loss/).

## Calibration and split controls: Appendix C

Each command runs independently. The first example evaluates 500 randomly initialized ResNet-18 models on both random CIFAR partitions and the original training/test split:

```bash
python controls.py cifar-vrt --arch r18_c10 --data-root datasets \
  --reps 0-499 --out_dir results/05_type_i_calibration_and_controls_before_training/records/cifar/vrt
python controls.py cifar-baselines --arch r18_c10 --data-root datasets \
  --part gpu --splits resplit,official --reps 0-499 \
  --out_dir results/05_type_i_calibration_and_controls_before_training/records/cifar/baselines
python controls.py cifar-baselines --arch r18_c10 --data-root datasets \
  --part cpu --splits resplit,official --reps 0-499 \
  --out_dir results/05_type_i_calibration_and_controls_before_training/records/cifar/baselines
python controls.py cifar-rule --arch r18_c10 --data-root datasets \
  --nets 0-29 --seeds 42-91 --out_dir results/05_type_i_calibration_and_controls_before_training/records/cifar/decision_rule
```

Repeat with the other architectures listed by `--help`. `cifar-official` runs the separate pixel and shuffled-extraction controls; `imagenet-vrt` runs the original ImageNet random-network control. The GPU/CPU baseline groups retain the paper's implementation choices.

These control outputs use the same record schemas as the supplied data. Once all required CIFAR settings are present under `results/`, `python results.py type-i --data results` summarizes them. A single architecture is a partial experiment, not the complete paper table.

## Synthetic sensitivity: Section 4.4 and Appendix D

```bash
python synthetic.py clusters --reps 200 --jobs 4 \
  --out results/06_synthetic_power_profile_details/records/fig4a_clusters.csv
python synthetic.py single-cluster --reps 200 --jobs 4 \
  --out results/06_synthetic_power_profile_details/records/fig4a_matched.csv
python synthetic.py bandwidth --reps 200 --jobs 4 \
  --out results/06_synthetic_power_profile_details/records/mmd_bandwidth.csv
python results.py synthetic --data results
```

The defaults use dimension 500, 5,000 generated points per split and 2,000 permutations. Cluster experiments cover 10 or 100 centres; single-cluster experiments include concentration, direction and vector-length changes. `--smoke` runs a much smaller execution check and does not reproduce the paper's power estimates.

For the reference-direction comparison in Appendix G:

```bash
python synthetic.py reference --reps 500 --jobs 4 \
  --out results/reference_direction.csv
```

The output records each condition and both reference choices separately. See [Scope](#scope) for the available original inputs.

## Score and class statistics: Appendices I–J

`mmd-scores` in `evaluate.py test` compares the VRT score distributions using MMD. The other recorded score statistics can be computed independently:

```bash
python evaluate.py score-statistics \
  --features 'features/r50_eval/ep{ep:03d}.npz' --model r50 \
  --epochs 0-200 --seeds 42-91 --permutations 2000 \
  --out results/r50_scores.csv
```

For classwise geometry, provide labels in the same row order as the feature arrays. The label NPZ uses `y_tr` and `y_te`:

```bash
python evaluate.py geometry \
  --features 'features/r50_eval/ep{ep:03d}.npz' --labels labels.npz \
  --model r50 --epochs 0-200 --out results/r50_geometry.csv
```

## CPU runtime: Appendix K

```bash
python evaluate.py timing \
  --features features/r50_eval/ep061.npz --model r50 --epoch 61 \
  --budget 10000 --seeds 42-51 --threads 8 --out results/r50_timing.csv
```

Repeat with `--budget 1500` and the ResNet20-FRN epoch-38 features. Timings exclude feature extraction and normalization; exact durations depend on the CPU. `results.py runtime` reads the saved float64 measurements used by Table 9.

## Result export

The included `data/` is already exported and validated. To clean another result tree using the historical schemas:

```bash
python results.py export --src raw_results --dst clean_results
```

The destination must be new or empty. Export from historical folders writes the current paper-section layout. The exporter validates records, rejects conflicting keys, removes mechanical file corruption and records cleaning and file hashes in `MANIFEST.csv`. It does not recompute p-values, except for recovering the recorded exact permutation-p columns. Fresh `evaluate.py` CSVs use `results.py new-runs` instead.

## Execution checks

After installing `requirements.txt`:

```bash
python -m pytest tests -q
python evaluate.py smoke
```

## Scope

- The released numerical results reproduce the table summaries without checkpoints. Original image datasets, model checkpoints and extracted feature arrays are not included.
- [figures/](figures/README.md) contains plotting scripts and additional numerical inputs, including matched synthetic repetitions and loss records. Generated figures are written to `results/figures/`, which is ignored by Git.
- Appendix E's full two-dimensional sweep outputs, runner and plotting script are not included. The [hyperparameter data guide](data/07_sensitivity_to_hyperparameters_and_sample_size/README.md#hyperparameter-sweeps-k-and-m) links the saved one-dimensional sweep logs. The original uniform-sphere calibration outputs are also unavailable here.
- Appendix G's reference-direction experiment can be rerun with `synthetic.py reference`; its original 500-repetition output is unavailable, so the included plot build reconstructs the figure from its saved vector asset. The extra 10-seed ImageNet diagnostics in Appendix H (class balancing, enlarged reference/query sets and test-vs-test controls) have no dedicated runner in the supplied source.
- Class-geometry measurements can be rerun with `evaluate.py geometry`, but the original Appendix J extraction inputs and plotting generator are not bundled.

Training and statistical seeds serve different purposes; rerunning training is not expected to recover bit-identical checkpoints or onset epochs.
