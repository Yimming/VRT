# SupContrast

The files in this folder (`losses.py`, `main_supcon.py`, `main_ce.py`, `main_linear.py`, `networks/`, `resnet_big.py`,
`util.py`, `tensorboard_logger.py`) are the SimCLR / SupCon training code of Yonglong Tian,
https://github.com/HobbitLong/SupContrast, under the BSD 2-Clause License reproduced in `LICENSE`. They are the
reference the SimCLR models of the paper were trained with (`code/train_simclr_supcon.py` follows `main_supcon.py`
and saves a checkpoint after every epoch) and `code/ssl_nll.py` evaluates the contrastive objective with the verbatim `SupConLoss` copy in
`train_simclr_supcon.py`, falling back to `losses.SupConLoss` from here. The copy was taken as used on the cluster and is not modified
in this repository; please check the upstream repository for its current version.
