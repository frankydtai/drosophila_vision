def dsi_cost_parts(pack, trace, bias_gt, *, b_offset=0, eps=1e-8):
    """Peak baseline-relative PD/ND responses, DSI target 1, squared error.

    Align entries to each hex's first stimulus, subtract baseline, and average
    curves within each cell/contrast/width/direction group before taking peaks.
    Use the intersection of the entries' expanded transit windows so every
    averaged sample includes every node. Clamp the resulting peaks to zero.

    DSI = (R_pd - R_nd) / (R_pd + R_nd + eps), matching NIPS2026
    ``direction_selectivity_loss`` (eps keeps grads finite near zero response).
    """
    if pack.dsi_t0s is None or pack.dsi_n_ts is None:
        raise ValueError("moving-bar DSI requires per-entry transit windows")
    stops = pack.dsi_t0s + pack.dsi_n_ts
    if bool((pack.dsi_t0s < 0).any()) or bool((stops > trace.shape[1]).any()):
        raise ValueError("moving-bar DSI window is outside the simulated trace")
    # hex_dsi_window has one transit duration before first stimulus and two
    # after it: length = 3 * duration + 1 (both endpoints included).
    before = (pack.dsi_n_ts - 1) // 3
    first_sti = pack.dsi_t0s + before
    offsets = torch.arange(
        -int(before.max()), int((pack.dsi_n_ts - 1 - before).max()) + 1,
        device=trace.device,
    )
    ts = first_sti[:, None] + offsets[None, :]
    valid = (ts >= pack.dsi_t0s[:, None]) & (ts < stops[:, None])
    response = trace[
        pack.entry_bs[:, None] + int(b_offset),
        ts.clamp(min=0, max=trace.shape[1] - 1),
        pack.entry_nodes[:, None],
    ] - bias_gt[:, None]
    parts = {}
    for key, pd, nd in dsi_groups(pack):
        pd_mean = response[pd].mean(dim=0)
        nd_mean = response[nd].mean(dim=0)
        r_pd = pd_mean.masked_fill(~valid[pd].all(dim=0), -torch.inf).amax().clamp_min(0.0)
        r_nd = nd_mean.masked_fill(~valid[nd].all(dim=0), -torch.inf).amax().clamp_min(0.0)
        dsi = (r_pd - r_nd) / (r_pd + r_nd + eps)
        parts[key] = (dsi - 1.0).square()
    return parts

