"""Visualise L4's highpass-as-(original - lowpass) decay.
Reuses the SAME filter functions and signal construction as
Medulla_Library.read_RecF_data (no re-implementation)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import blindschleiche_py3 as bs

# --- identical step signal to read_RecF_data ---
signal = np.zeros(200)
signal[50:200] = 1.0
signal = bs.lowpass(signal, 5)
signal = signal / np.max(signal)

# --- L4 (cell index 3) time constants (tau_hp overridable via argv) ---
IR_hp = float(sys.argv[1]) if len(sys.argv) > 1 else 38.1
IR_lp = 2.3

# bandpass(signal, hp, lp) = highpass(lowpass(signal, lp), hp)
x = bs.lowpass(signal, IR_lp)        # signal entering the highpass stage ("原始")
lp = bs.lowpass(x, IR_hp)            # its lowpass component ("低通", tau_hp)
hp = x - lp                          # subtraction = highpass output ("相減") = L4 final

t = np.arange(200)
plt.figure(figsize=(9, 5))
plt.plot(t, x,  color="tab:blue",   lw=2, label=r"original  $x=\mathrm{lowpass}(signal,\ \tau_{lp}=2.3)$")
plt.plot(t, lp, color="tab:orange", lw=2, ls="--", label=r"lowpass  $\mathrm{lowpass}(x,\ \tau_{hp}=%.1f)$" % IR_hp)
plt.plot(t, hp, color="tab:red",    lw=2.5, label=r"subtraction  $x-\mathrm{lowpass}(x)$  = L4 highpass out")
plt.axhline(0, color="k", lw=0.6)
plt.axvline(50, color="gray", lw=0.6, ls=":")
plt.title(r"L4: highpass = original - lowpass ($\tau_{hp}=%.1f$) -> steady state removed, decays to 0" % IR_hp)
plt.xlabel("time step (x10 ms)")
plt.ylabel("amplitude")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "l4_highpass_decomp_hp%g.png" % IR_hp)
plt.savefig(out, dpi=130)
print("saved ->", out)
