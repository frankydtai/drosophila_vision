"""Conductance: per-cell out_scale (65, 'individual') vs one shared out_scale ('shared').

out_scale is now an official schema parameter (65 per-cell by default). This test
just flips its mode via the same CLI/SLURM mechanism (run_training param_modes) and
trains each variant through the EXACT run.py pipeline, so each lands in its own
run_<id> folder with the full standard output set (no rewriting of run.py / fc).
"""
import exp_utils as E
import run

fc = E.fc
LRS = list(E.LRS)
NRUNS = 1
STEPS = E.STEPS


if __name__ == "__main__":
    f1, d1 = run.run_training("conductance", NRUNS, STEPS, LRS,
                              fname="training_outscale_individual.npy",
                              param_modes={"out_scale": "individual"})
    f2, d2 = run.run_training("conductance", NRUNS, STEPS, LRS,
                              fname="training_outscale_shared.npy",
                              param_modes={"out_scale": "shared"})
    print("\n=== conductance out_scale: 65 per-cell (individual) vs 1 shared ===")
    print("  individual (65) -> %s/%s" % (d1, f1))
    print("  shared     (1)  -> %s/%s" % (d2, f2))
