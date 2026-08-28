#!/usr/bin/env python3
"""Train PCGCv2 on explicit train/calibration directories."""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    upstream = args.upstream_dir.resolve()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    import numpy as np
    import torch
    from data_loader import PCDataset, make_data_loader
    from pcc_model import PCCModel
    from train import TrainingConfig
    from trainer import Trainer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_files = sorted(glob.glob(str(args.train_dir.resolve() / "*.ply")))
    calibration_files = sorted(
        glob.glob(str(args.calibration_dir.resolve() / "*.ply"))
    )
    if not train_files or not calibration_files:
        raise RuntimeError("explicit PCGCv2 train/calibration split is empty")
    if set(train_files) & set(calibration_files):
        raise RuntimeError("PCGCv2 train and calibration files overlap")
    output = args.output_dir.resolve()
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    training = TrainingConfig(
        logdir=str(log_dir),
        ckptdir=str(output),
        init_ckpt="",
        alpha=args.alpha,
        beta=args.beta,
        lr=args.learning_rate,
        check_time=10,
    )
    trainer = Trainer(config=training, model=PCCModel())
    train_loader = make_data_loader(
        PCDataset(train_files), batch_size=args.batch_size, shuffle=True, repeat=False
    )
    calibration_loader = make_data_loader(
        PCDataset(calibration_files),
        batch_size=args.batch_size,
        shuffle=False,
        repeat=False,
    )
    for epoch in range(args.epochs):
        if epoch:
            trainer.config.lr = max(trainer.config.lr / 2, 1e-5)
        trainer.train(train_loader)
        trainer.test(calibration_loader, "Calibration")


if __name__ == "__main__":
    main()
