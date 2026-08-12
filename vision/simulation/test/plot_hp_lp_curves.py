"""Plot isolated hp_lp membrane on a pulse (no network).

Uses ``neuron.model_hp_lp.update_state_hp_lp`` (same Euler as training):

    τ_HP d v_slow / dt = v_drive − v_slow
    v_hp = v_drive − a_slow v_slow
    τ_lp dv/dt = −(v − v_rest) + v_hp

with v_drive = v_sti (v_in = 0; no v_rest in HP path), v_sti = i_sti / g_leak.
Time indexing matches ``neuron.forward.forward_full``: v[0] from init;
v[t] uses i_sti[t-1].

LP-only row uses a large τ_HP (HP effectively off).

Rows: input | HP stage | LP-only vs HP→LP | sweep a_slow | sweep τ_HP | sweep τ_lp | sweep pulse.
Columns: left = +pulse, right = −pulse.

Usage (from ``vision/simulation/``):

    ../.venv/bin/python test/plot_hp_lp_curves.py
    ../.venv/bin/python test/plot_hp_lp_curves.py --show
    ../.venv/bin/python test/plot_hp_lp_curves.py --euler ex --hp-a-slow-list 0.1,0.5,1
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

from figure.util import save_figure
from import_bootstrap import parse_comma_list
from neuron.model_hp_lp import update_state_hp_lp
from neuron.param import expand_euler
from default_params import EULER, G_LEAK, STATE_CLAMP

DEFAULT_SAVE = os.path.join(HERE, "hp_lp_curves.png")
DEFAULT_A_SLOW = 1.0
DEFAULT_A_SLOW_LIST = "0,0.5,1,1.5"
DEFAULT_HP_A_SLOW_LIST = "0.1,0.5,1"
DEFAULT_TAU_HP_LIST = "80,200,500,2000"
DEFAULT_TAU_LP_LIST = "20,50,100"
DEFAULT_PULSE_LIST = "50,100,500"
TAU_HP_OFF_MS = 1.0e6

_BACKEND = SimpleNamespace(
    n_nodes=1,
    conn=SimpleNamespace(
        signed_drive=lambda x, syn_strength: torch.zeros_like(x),
        exc_inh_drive=lambda x, syn_strength: (
            torch.zeros_like(x),
            torch.zeros_like(x),
        ),
    ),
)


def make_pulse(t_ms: np.ndarray, *, t_on_ms: float, ms_pulse: float, s0: float) -> np.ndarray:
    t_off_ms = t_on_ms + ms_pulse
    return np.where((t_ms >= t_on_ms) & (t_ms < t_off_ms), s0, 0.0).astype(np.float64)


def _p_tensors(*, v_rest, tau_lp_ms, tau_hp_ms, a_slow):
    z = torch.zeros(1, dtype=torch.float32)
    one = torch.ones(1, dtype=torch.float32)
    return {
        "v_rest": torch.tensor([float(v_rest)], dtype=torch.float32),
        "tau_lp": torch.tensor([float(tau_lp_ms)], dtype=torch.float32),
        "tau_hp": torch.tensor([float(tau_hp_ms)], dtype=torch.float32),
        "a_slow": torch.tensor([float(a_slow)], dtype=torch.float32),
        "a_in": one.clone(),
        "a_out": one.clone(),
        "v_th": z.clone(),
        "syn_strength_cell": one.clone(),
    }


def simulate_hp_lp(
    v_sti: np.ndarray,
    *,
    tau_lp_ms: float,
    tau_hp_ms: float,
    v_rest: float,
    a_slow: float,
    delta_ms: float,
    g_leak: float,
    state_clamp: float,
    euler: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (v, v_slow, v_hp); time index matches ``forward_full``."""
    euler = expand_euler(euler)
    g_leak = float(g_leak)
    if g_leak == 0.0:
        raise ValueError("g_leak must be non-zero")
    x = np.asarray(v_sti, dtype=np.float64)
    i_sti = torch.as_tensor(x * g_leak, dtype=torch.float32)
    t_end = int(i_sti.numel())
    p = _p_tensors(
        v_rest=v_rest, tau_lp_ms=tau_lp_ms, tau_hp_ms=tau_hp_ms, a_slow=a_slow,
    )
    # pre_steady: v_slow0 = v_drive0, v0 = v_rest + v_drive0 − a_slow v_slow0
    v_drive0 = i_sti[0] / g_leak
    v_slow = v_drive0.view(1, 1).clone()
    v = (p["v_rest"] + v_drive0 - p["a_slow"] * v_slow.view(-1)).view(1, 1).clone()
    v_out = np.empty(t_end, dtype=np.float64)
    slow_out = np.empty(t_end, dtype=np.float64)
    hp_out = np.empty(t_end, dtype=np.float64)
    v_out[0] = float(v.item())
    slow_out[0] = float(v_slow.item())
    hp_out[0] = float((v_drive0 - p["a_slow"] * v_slow.view(-1)).item())
    for t in range(1, t_end):
        v, v_slow, comp = update_state_hp_lp(
            v, v_slow, p, i_sti[t - 1].view(1, 1), _BACKEND,
            delta_ms=delta_ms, state_clamp=state_clamp, g_leak=g_leak, euler=euler,
            return_component=True,
        )
        v_out[t] = float(v.item())
        slow_out[t] = float(comp["v_slow"].item())
        hp_out[t] = float(comp["v_hp"].item())
    return v_out, slow_out, hp_out


def _shade_and_grid(ax, t_on_s: float, t_off_s: float, v_rest: float) -> None:
    ax.axvspan(t_on_s, t_off_s, color="0.92", zorder=0)
    ax.axhline(v_rest, color="0.5", lw=0.6, ls=":")
    ax.grid(True, alpha=0.25)


def _cmap_colors(n: int):
    cmap = plt.cm.viridis
    if n <= 1:
        return [cmap(0.55)]
    return [cmap(0.15 + 0.75 * (i / (n - 1))) for i in range(n)]


def _fill_column(
    axes_col,
    *,
    t_ms: np.ndarray,
    t_s: np.ndarray,
    t_on_ms: float,
    ms_pulse: float,
    s0: float,
    v_rest: float,
    a_slow: float,
    tau_lp_ms: float,
    tau_hp_ms: float,
    dt_ms: float,
    g_leak: float,
    state_clamp: float,
    euler: str,
    a_slow_list: list[float],
    hp_a_slow_list: list[float],
    tau_hp_list: list[float],
    tau_lp_list: list[float],
    pulse_list: list[float],
    show_ylabel: bool,
    show_legend: bool,
    col_title: str,
) -> None:
    ax0, ax1, ax2, ax3, ax4, ax5, ax6 = axes_col
    t_on_s = t_on_ms / 1000.0
    t_off_s = (t_on_ms + ms_pulse) / 1000.0
    v_sti = make_pulse(t_ms, t_on_ms=t_on_ms, ms_pulse=ms_pulse, s0=s0)
    sim_kw = dict(
        v_rest=v_rest, a_slow=a_slow, delta_ms=dt_ms,
        g_leak=g_leak, state_clamp=state_clamp, euler=euler,
    )

    v_lp, _, _ = simulate_hp_lp(
        v_sti, tau_lp_ms=tau_lp_ms, tau_hp_ms=TAU_HP_OFF_MS, **sim_kw,
    )
    v_hp_lp, v_slow, _ = simulate_hp_lp(
        v_sti, tau_lp_ms=tau_lp_ms, tau_hp_ms=tau_hp_ms, **sim_kw,
    )

    for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
        _shade_and_grid(ax, t_on_s, t_off_s, v_rest if ax in (ax2, ax3, ax4, ax5) else 0.0)
    ax6.axhline(v_rest, color="0.5", lw=0.6, ls=":")
    ax6.axvline(t_on_s, color="0.6", lw=0.7, ls="--")
    ax6.grid(True, alpha=0.25)

    ax0.plot(t_s, v_sti, color="0.15", lw=1.6, label=r"$v_{\mathrm{sti}}$")
    ax0.set_title(col_title, fontsize=10)
    if show_legend:
        ax0.legend(loc="upper right", fontsize=7, frameon=False)

    ax1.plot(t_s, v_slow, color="0.35", lw=1.2, ls="--", label=r"$v_{\mathrm{slow}}$")
    for a, color in zip(hp_a_slow_list, _cmap_colors(len(hp_a_slow_list))):
        _, _, v_hp = simulate_hp_lp(
            v_sti, tau_lp_ms=tau_lp_ms, tau_hp_ms=tau_hp_ms,
            v_rest=v_rest, a_slow=a, delta_ms=dt_ms,
            g_leak=g_leak, state_clamp=state_clamp, euler=euler,
        )
        ax1.plot(
            t_s, v_hp, color=color, lw=1.5,
            label=rf"$v_{{\mathrm{{hp}}}}$ ($a_{{\mathrm{{slow}}}}$={a:g})",
        )
    ax1.set_title(
        rf"HP stage sweep $a_{{\mathrm{{slow}}}}$ "
        rf"($\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms)",
        fontsize=8,
    )
    if show_legend:
        ax1.legend(loc="upper right", fontsize=6, frameon=False)

    ax2.plot(t_s, v_lp, color="C2", lw=1.8, label=rf"LP only ($\tau_{{\mathrm{{HP}}}}$={TAU_HP_OFF_MS:g})")
    ax2.plot(t_s, v_hp_lp, color="C3", lw=1.8, label="HP → LP")
    if show_legend:
        ax2.legend(loc="upper right", fontsize=7, frameon=False)

    for a, color in zip(a_slow_list, _cmap_colors(len(a_slow_list))):
        v, _, _ = simulate_hp_lp(
            v_sti, tau_lp_ms=tau_lp_ms, tau_hp_ms=tau_hp_ms,
            v_rest=v_rest, a_slow=a, delta_ms=dt_ms,
            g_leak=g_leak, state_clamp=state_clamp, euler=euler,
        )
        ax3.plot(t_s, v, color=color, lw=1.6, label=rf"$a_{{\mathrm{{slow}}}}$={a:g}")
    ax3.set_title(
        rf"sweep $a_{{\mathrm{{slow}}}}$ ($\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms, "
        rf"$\tau_{{\mathrm{{lp}}}}$={tau_lp_ms:g} ms)",
        fontsize=8,
    )
    if show_legend:
        ax3.legend(loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(a_slow_list)))

    for thp, color in zip(tau_hp_list, _cmap_colors(len(tau_hp_list))):
        v, _, _ = simulate_hp_lp(
            v_sti, tau_lp_ms=tau_lp_ms, tau_hp_ms=thp, **sim_kw,
        )
        ax4.plot(t_s, v, color=color, lw=1.6, label=rf"$\tau_{{\mathrm{{HP}}}}$={thp:g}")
    ax4.set_title(
        rf"sweep $\tau_{{\mathrm{{HP}}}}$ ($a_{{\mathrm{{slow}}}}$={a_slow:g}, "
        rf"$\tau_{{\mathrm{{lp}}}}$={tau_lp_ms:g} ms)",
        fontsize=8,
    )
    if show_legend:
        ax4.legend(loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(tau_hp_list)))

    for tlp, color in zip(tau_lp_list, _cmap_colors(len(tau_lp_list))):
        v, _, _ = simulate_hp_lp(
            v_sti, tau_lp_ms=tlp, tau_hp_ms=tau_hp_ms, **sim_kw,
        )
        ax5.plot(t_s, v, color=color, lw=1.6, label=rf"$\tau_{{\mathrm{{lp}}}}$={tlp:g}")
    ax5.set_title(
        rf"sweep $\tau_{{\mathrm{{lp}}}}$ ($a_{{\mathrm{{slow}}}}$={a_slow:g}, "
        rf"$\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms)",
        fontsize=8,
    )
    if show_legend:
        ax5.legend(loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(tau_lp_list)))

    for pw, color in zip(pulse_list, _cmap_colors(len(pulse_list))):
        v_pw = make_pulse(t_ms, t_on_ms=t_on_ms, ms_pulse=pw, s0=s0)
        v, _, _ = simulate_hp_lp(
            v_pw, tau_lp_ms=tau_lp_ms, tau_hp_ms=tau_hp_ms, **sim_kw,
        )
        ax6.axvspan(t_on_s, (t_on_ms + pw) / 1000.0, color=color, alpha=0.08, zorder=0)
        ax6.plot(t_s, v, color=color, lw=1.6, label=rf"$T$={pw:g} ms")
    ax6.set_title(
        rf"sweep pulse ($a_{{\mathrm{{slow}}}}$={a_slow:g}, $\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms, "
        rf"$\tau_{{\mathrm{{lp}}}}$={tau_lp_ms:g} ms)",
        fontsize=8,
    )
    ax6.set_xlabel("time (s)")
    if show_legend:
        ax6.legend(loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(pulse_list)))

    if show_ylabel:
        ax0.set_ylabel(r"$v_{\mathrm{sti}}$")
        ax1.set_ylabel("HP stage")
        ax2.set_ylabel(r"$v$")
        ax3.set_ylabel(r"$v$ (sweep $a_{\mathrm{slow}}$)")
        ax4.set_ylabel(r"$v$ (sweep $\tau_{\mathrm{HP}}$)")
        ax5.set_ylabel(r"$v$ (sweep $\tau_{\mathrm{lp}}$)")
        ax6.set_ylabel(r"$v$ (sweep pulse)")


def plot_hp_lp(
    path: str,
    *,
    show: bool = False,
    dt_ms: float = 1.0,
    t_total_ms: float = 3000.0,
    t_on_ms: float = 500.0,
    ms_pulse: float = 1500.0,
    s0: float = 1.0,
    v_rest: float = 0.0,
    a_slow: float = DEFAULT_A_SLOW,
    tau_lp_ms: float = 50.0,
    tau_hp_ms: float = 200.0,
    g_leak: float = float(G_LEAK),
    state_clamp: float = float(STATE_CLAMP),
    euler: str = EULER,
    a_slow_list: list[float] | None = None,
    hp_a_slow_list: list[float] | None = None,
    tau_hp_list: list[float] | None = None,
    tau_lp_list: list[float] | None = None,
    pulse_list: list[float] | None = None,
) -> None:
    if a_slow_list is None:
        a_slow_list = [float(x) for x in parse_comma_list(DEFAULT_A_SLOW_LIST)]
    if hp_a_slow_list is None:
        hp_a_slow_list = [float(x) for x in parse_comma_list(DEFAULT_HP_A_SLOW_LIST)]
    if tau_hp_list is None:
        tau_hp_list = [float(x) for x in parse_comma_list(DEFAULT_TAU_HP_LIST)]
    if tau_lp_list is None:
        tau_lp_list = [float(x) for x in parse_comma_list(DEFAULT_TAU_LP_LIST)]
    if pulse_list is None:
        pulse_list = [float(x) for x in parse_comma_list(DEFAULT_PULSE_LIST)]
    euler = expand_euler(euler)

    n = int(round(t_total_ms / dt_ms)) + 1
    t_ms = np.arange(n, dtype=np.float64) * dt_ms
    t_s = t_ms / 1000.0

    fig, axes = plt.subplots(
        7,
        2,
        figsize=(12.0, 14.5),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    common = dict(
        t_ms=t_ms,
        t_s=t_s,
        t_on_ms=t_on_ms,
        ms_pulse=ms_pulse,
        v_rest=v_rest,
        a_slow=a_slow,
        tau_lp_ms=tau_lp_ms,
        tau_hp_ms=tau_hp_ms,
        dt_ms=dt_ms,
        g_leak=g_leak,
        state_clamp=state_clamp,
        euler=euler,
        a_slow_list=a_slow_list,
        hp_a_slow_list=hp_a_slow_list,
        tau_hp_list=tau_hp_list,
        tau_lp_list=tau_lp_list,
        pulse_list=pulse_list,
    )
    param_line = (
        rf"pulse={ms_pulse:g} ms, $\tau_{{\mathrm{{lp}}}}$={tau_lp_ms:g} ms, "
        rf"$\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms, $a_{{\mathrm{{slow}}}}$={a_slow:g}, "
        rf"$V_{{\mathrm{{rest}}}}$={v_rest:g}, $|v_{{\mathrm{{sti}}}}|$={abs(s0):g}, "
        rf"euler={euler}, $\Delta t$={dt_ms:g} ms"
    )
    _fill_column(
        axes[:, 0],
        s0=abs(s0),
        show_ylabel=True,
        show_legend=True,
        col_title=rf"+pulse ($v_{{\mathrm{{sti}}}}$={abs(s0):g})",
        **common,
    )
    _fill_column(
        axes[:, 1],
        s0=-abs(s0),
        show_ylabel=False,
        show_legend=False,
        col_title=rf"−pulse ($v_{{\mathrm{{sti}}}}$={-abs(s0):g})",
        **common,
    )
    axes[0, 0].annotate(
        param_line,
        xy=(0.0, 1.18),
        xycoords="axes fraction",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    fig.suptitle(
        "hp_lp isolated membrane: LP-only vs HP→LP (+pulse | −pulse)",
        fontsize=12,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    save_figure(fig, path, dpi=150)
    print(f"saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", default=DEFAULT_SAVE, help="output PNG path")
    p.add_argument("--show", action="store_true")
    p.add_argument("--dt-ms", type=float, default=1.0)
    p.add_argument("--t-total-ms", type=float, default=3000.0)
    p.add_argument("--t-on-ms", type=float, default=500.0)
    p.add_argument("--ms-pulse", type=float, default=1500.0)
    p.add_argument("--s0", type=float, default=1.0, help="pulse |v_sti| (right column uses −|v_sti|)")
    p.add_argument("--v-rest", type=float, default=0.0)
    p.add_argument("--a-slow", type=float, default=DEFAULT_A_SLOW, help="baseline a_slow")
    p.add_argument("--tau-lp", type=float, default=50.0, help="baseline membrane tau_lp [ms]")
    p.add_argument("--tau-hp", type=float, default=200.0, help="baseline tau_hp [ms]")
    p.add_argument("--g-leak", type=float, default=float(G_LEAK))
    p.add_argument("--euler", default=EULER, choices=("im", "ex", "implicit", "explicit"))
    p.add_argument("--a-slow-list", type=str, default=DEFAULT_A_SLOW_LIST, help="comma-separated a_slow sweep (row v)")
    p.add_argument(
        "--hp-a-slow-list", type=str, default=DEFAULT_HP_A_SLOW_LIST,
        help="comma-separated a_slow for HP-stage v_hp sweep (row 2)",
    )
    p.add_argument("--tau-hp-list", type=str, default=DEFAULT_TAU_HP_LIST, help="comma-separated τ_HP sweep [ms]")
    p.add_argument("--tau-lp-list", type=str, default=DEFAULT_TAU_LP_LIST, help="comma-separated τ_lp sweep [ms]")
    p.add_argument("--pulse-list", type=str, default=DEFAULT_PULSE_LIST, help="comma-separated pulse widths [ms]")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for name, val in (
        ("--dt-ms", args.dt_ms),
        ("--t-total-ms", args.t_total_ms),
        ("--ms-pulse", args.ms_pulse),
        ("--tau-lp", args.tau_lp),
        ("--tau-hp", args.tau_hp),
        ("--g-leak", args.g_leak),
    ):
        if val <= 0:
            raise ValueError(f"{name} must be > 0")
    a_slow_list = [float(x) for x in parse_comma_list(args.a_slow_list)]
    hp_a_slow_list = [float(x) for x in parse_comma_list(args.hp_a_slow_list)]
    tau_hp_list = [float(x) for x in parse_comma_list(args.tau_hp_list)]
    tau_lp_list = [float(x) for x in parse_comma_list(args.tau_lp_list)]
    pulse_list = [float(x) for x in parse_comma_list(args.pulse_list)]
    if not a_slow_list:
        raise ValueError("--a-slow-list must be non-empty")
    if not hp_a_slow_list:
        raise ValueError("--hp-a-slow-list must be non-empty")
    if not tau_hp_list:
        raise ValueError("--tau-hp-list must be non-empty")
    if not tau_lp_list:
        raise ValueError("--tau-lp-list must be non-empty")
    if not pulse_list:
        raise ValueError("--pulse-list must be non-empty")
    if any(a < 0 for a in a_slow_list):
        raise ValueError("--a-slow-list values must be >= 0")
    if any(a < 0 for a in hp_a_slow_list):
        raise ValueError("--hp-a-slow-list values must be >= 0")
    if any(t <= 0 for t in tau_hp_list):
        raise ValueError("--tau-hp-list values must be > 0")
    if any(t <= 0 for t in tau_lp_list):
        raise ValueError("--tau-lp-list values must be > 0")
    if any(t <= 0 for t in pulse_list):
        raise ValueError("--pulse-list values must be > 0")
    plot_hp_lp(
        args.save,
        show=args.show,
        dt_ms=args.dt_ms,
        t_total_ms=args.t_total_ms,
        t_on_ms=args.t_on_ms,
        ms_pulse=args.ms_pulse,
        s0=args.s0,
        v_rest=args.v_rest,
        a_slow=args.a_slow,
        tau_lp_ms=args.tau_lp,
        tau_hp_ms=args.tau_hp,
        g_leak=args.g_leak,
        euler=args.euler,
        a_slow_list=a_slow_list,
        hp_a_slow_list=hp_a_slow_list,
        tau_hp_list=tau_hp_list,
        tau_lp_list=tau_lp_list,
        pulse_list=pulse_list,
    )


if __name__ == "__main__":
    main()
