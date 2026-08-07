#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end pipeline demo for neuro_framework.

Pipeline:
  1. Load connectome  (ConnectomeLoader)
  2. Build network    (ConnectomeNetwork)
  3. Generate stimulus(build_stimulus_tensor / hex_grid_coords)
  4. Simulate         (net.simulate)
  5. Compute loss     (combined_loss)
  6. Train            (TorchTrainer)
  7. Print summary

Run from repo root:
    .venv/bin/python -m neuro_framework.run_pipeline
or:
    .venv/bin/python neuro_framework/run_pipeline.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Step 1 — Load connectome
# ---------------------------------------------------------------------------
from neuro_framework.connectome.loader import ConnectomeLoader


def step1_load_connectome(source: str = "banc") -> ConnectomeLoader:
    log.info("=== Step 1: Loading connectome (source=%s) ===", source)
    if source == "banc":
        loader = ConnectomeLoader.from_banc(cell_types=["T4a", "T4b", "T4c", "T4d"])
    elif source == "fafb":
        loader = ConnectomeLoader.from_fafb()
    elif source == "flyvis":
        loader = ConnectomeLoader.from_flyvis()
    elif source == "optic_lobe":
        loader = ConnectomeLoader.from_optic_lobe()
    else:
        raise ValueError(f"Unknown source '{source}'.")

    nodes, edges = loader.load()
    s = loader.summary()
    log.info("  source       : %s", s["source"])
    log.info("  n_neurons    : %d", s["n_neurons"])
    log.info("  n_edges      : %d", s["n_edges"])
    log.info("  n_cell_types : %d", s["n_cell_types"])
    log.info("  NT dist      : %s", s["nt_distribution"])
    log.info("  nodes sample :\n%s", nodes.head(3).to_string())
    log.info("  edges sample :\n%s", edges.head(3).to_string())
    return loader


# ---------------------------------------------------------------------------
# Step 2 — Build network
# ---------------------------------------------------------------------------
from neuro_framework.models.network_torch import ConnectomeNetwork


def step2_build_network(loader: ConnectomeLoader, dynamics: str = "lif") -> ConnectomeNetwork:
    log.info("=== Step 2: Building ConnectomeNetwork (dynamics=%s) ===", dynamics)
    net = ConnectomeNetwork.from_loader(loader, dynamics=dynamics)
    log.info("  %s", net)
    log.info("  Trainable params: %d", net.n_parameters())
    return net


# ---------------------------------------------------------------------------
# Step 3 — Generate stimulus
# ---------------------------------------------------------------------------
from neuro_framework.stimulus.visual import hex_grid_coords, build_stimulus_tensor


def step3_build_stimulus(
    n_nodes: int,
    T: int = 100,
    dt: float = 1.0,
    batch_size: int = 4,
    stimulus_type: str = "moving_bar",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    log.info(
        "=== Step 3: Generating stimulus (type=%s, T=%d ms, batch=%d) ===",
        stimulus_type, T, batch_size,
    )
    # Smallest hex radius whose node count >= n_nodes
    radius = 1
    while (3 * radius * (radius - 1) + 1) < n_nodes:
        radius += 1
    coords = hex_grid_coords(radius)   # (n_hex, 2)

    x_full = build_stimulus_tensor(
        stimulus_type,
        coords=coords,
        batch_size=batch_size,
        device=device,
        t_start=0.0,
        t_end=float(T),
        dt=dt,
    )   # (batch, T, n_hex)

    n_hex = x_full.shape[2]
    if n_hex >= n_nodes:
        x = x_full[:, :, :n_nodes]
    else:
        pad = torch.zeros(batch_size, T, n_nodes - n_hex, device=device)
        x = torch.cat([x_full, pad], dim=2)

    log.info("  stimulus shape : %s  (batch, T, n_nodes)", tuple(x.shape))
    log.info("  min/max/mean   : %.3f / %.3f / %.3f",
             x.min().item(), x.max().item(), x.mean().item())
    return x


# ---------------------------------------------------------------------------
# Step 4 — Simulate (forward pass, no grad)
# ---------------------------------------------------------------------------
def step4_simulate(
    net: ConnectomeNetwork,
    x: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    log.info("=== Step 4: Forward simulation ===")
    net.eval()
    with torch.no_grad():
        activity = net.simulate(x, dt=dt)   # (batch, T, n_nodes)
    log.info("  activity shape : %s", tuple(activity.shape))
    log.info("  act min/max    : %.4f / %.4f",
             activity.min().item(), activity.max().item())
    log.info("  act mean/std   : %.4f / %.4f",
             activity.mean().item(), activity.std().item())
    return activity


# ---------------------------------------------------------------------------
# Step 5 — Loss check (pre-training sanity)
# ---------------------------------------------------------------------------
from neuro_framework.training.losses import (
    mse_loss,
    correlation_loss,
    direction_selectivity_loss,
    combined_loss,
)


def step5_check_loss(activity: torch.Tensor) -> None:
    log.info("=== Step 5: Loss check (pre-training) ===")
    batch, T, N = activity.shape
    n_out = min(20, N)
    pred   = activity[:, :, :n_out]
    target = torch.rand_like(pred) * 0.5    # synthetic Ca2+ traces

    losses = {
        "mse":  (mse_loss(pred, target), 1.0),
        "corr": (correlation_loss(pred, target), 0.5),
    }
    half = batch // 2
    if half > 0:
        losses["dsi"] = (
            direction_selectivity_loss(
                activity[:half, :, :n_out],
                activity[half:half*2, :, :n_out],
            ),
            0.1,
        )

    total, breakdown = combined_loss(losses)
    for name, val in breakdown.items():
        log.info("  %-10s = %.4f", name, val.item())
    log.info("  total_loss  = %.4f  (weighted)", total.item())


# ---------------------------------------------------------------------------
# Step 6 — Training loop
# ---------------------------------------------------------------------------
from neuro_framework.training.trainer import TorchTrainer


def step6_train(
    net: ConnectomeNetwork,
    x: torch.Tensor,
    n_epochs: int = 100,
    dt: float = 1.0,
) -> None:
    log.info("=== Step 6: Training (n_epochs=%d) ===", n_epochs)
    batch, T, N = x.shape
    n_out = min(20, N)

    # Synthetic calcium targets, zero-padded to n_nodes
    y_full = torch.zeros(batch, T, N, device=x.device)
    y_full[:, :, :n_out] = torch.rand(batch, T, n_out, device=x.device) * 0.5

    trainer = TorchTrainer(
        net,
        optimizer_cfg={"lr": 5e-4, "weight_decay": 1e-5},
    )

    split = max(1, batch // 2)
    x_train, y_train = x[:split], y_full[:split]
    x_val,   y_val   = x[split:], y_full[split:]

    history = trainer.train(
        x_train, y_train,
        n_epochs=n_epochs,
        loss_name="mse",
        x_val=x_val if x_val.shape[0] > 0 else None,
        y_val=y_val if y_val.shape[0] > 0 else None,
    )

    log.info("  Initial loss : %.4f", history.train_loss[0])
    log.info("  Final   loss : %.4f", history.train_loss[-1])
    if history.val_loss:
        log.info("  Final val    : %.4f", history.val_loss[-1])
    log.info("  Epochs run   : %d", len(history.train_loss))


# ---------------------------------------------------------------------------
# Step 7 — Summary
# ---------------------------------------------------------------------------
def step7_summary(loader: ConnectomeLoader, net: ConnectomeNetwork) -> None:
    log.info("=== Step 7: Summary ===")
    s = loader.summary()
    log.info("  Dataset      : %s", s["source"])
    log.info("  Neurons      : %d", s["n_neurons"])
    log.info("  Synapses     : %d", s["n_edges"])
    log.info("  Model params : %d", net.n_parameters())
    log.info("  Dynamics     : %s", net.dynamics.__class__.__name__)
    log.info("Pipeline completed successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(
    source: str = "banc",
    dynamics: str = "lif",
    T: int = 100,
    dt: float = 1.0,
    batch_size: int = 4,
    n_epochs: int = 10,
    stimulus_type: str = "moving_bar",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    loader   = step1_load_connectome(source=source)
    net      = step2_build_network(loader, dynamics=dynamics)
    net      = net.to(device)
    x        = step3_build_stimulus(
                   n_nodes=net.n_nodes, T=T, dt=dt,
                   batch_size=batch_size,
                   stimulus_type=stimulus_type,
                   device=device,
               )
    activity = step4_simulate(net, x, dt=dt)
    step5_check_loss(activity)
    step6_train(net, x, n_epochs=n_epochs, dt=dt)
    step7_summary(loader, net)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="neuro_framework end-to-end pipeline")
    parser.add_argument("--source",   default="banc",
                        choices=["banc", "fafb", "flyvis", "optic_lobe"])
    parser.add_argument("--dynamics", default="lif",
                        choices=["voltage", "dmn", "lif", "hh"])
    parser.add_argument("--T",        type=int,   default=100)
    parser.add_argument("--dt",       type=float, default=1.0)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--epochs",   type=int,   default=10)
    parser.add_argument("--stimulus", default="moving_bar",
                        choices=["flash", "moving_bar", "moving_edge", "grating"])
    args = parser.parse_args()

    main(
        source        = args.source,
        dynamics      = args.dynamics,
        T             = args.T,
        dt            = args.dt,
        batch_size    = args.batch,
        n_epochs      = args.epochs,
        stimulus_type = args.stimulus,
    )
