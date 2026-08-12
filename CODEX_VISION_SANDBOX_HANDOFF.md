# Codex Vision / Sandbox Handoff

This note is for a new Codex agent/session after restarting the Cursor/Codex
window.

## Current Problem

`view_image` cannot read local PNG files in the current session. Example failure:

```text
unable to locate image at `/home/yitai/drosophila_vision/figure_digitization/gruntman18/2b_digitized.png`:
fs sandbox helper failed with status exit status: 1:
bwrap: Creating new namespace failed: No space left on device
```

The image file exists. The error is from the sandbox helper, not from the PNG.

## Root Cause Found

This SSH/HPC host is RHEL 9.8 on `spartan-login2.hpc.unimelb.edu.au`.

The host allows normal user/mount namespaces, but disables user network
namespaces:

```text
user.max_user_namespaces = 12000
user.max_mnt_namespaces = 1027227
user.max_net_namespaces = 0
```

Direct tests showed:

```bash
bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp true
# succeeded

bwrap --unshare-net --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp true
# failed: bwrap: Creating new namespace failed: No space left on device

unshare -Ur true
# succeeded

unshare -n true
# failed: Operation not permitted
```

So the failure is caused by Codex's restricted sandbox trying to create a
network namespace on a host where unprivileged net namespaces are disabled.

## Config Change Already Made

The user explicitly approved the workaround:

```toml
sandbox_mode = "danger-full-access"
```

It was added at the top of:

```text
/home/yitai/.codex/config.toml
```

A backup was created before changing it:

```text
/home/yitai/.codex/config.toml.bak-20260812-sandbox
```

Do not dump the full config unless needed; it may contain provider credentials.

`codex doctor` after the change reported:

```text
sandbox: unrestricted fs + enabled network
```

That means the config is being read correctly by Codex CLI.

## Why It Still Failed In The Old Window

After changing config, the already-running conversation/tool runtime still used
the old managed sandbox. In that old thread:

```bash
ls
# still failed with bwrap ENOSPC
```

and:

```text
view_image(...)
# still failed with bwrap ENOSPC
```

The expected fix is to open a fresh Codex/Cursor chat/session so the tool host
is rebuilt with the new config.

## Tests For The New Agent

Run these first in the new session.

1. Check ordinary tool sandbox:

```bash
ls
```

Expected: should list repo files, not fail with `bwrap`.

2. Check Codex config:

```bash
codex doctor
```

Expected sandbox line:

```text
unrestricted fs + enabled network
```

3. Check visual multimodal local image reading:

Use the tool equivalent of:

```text
view_image("/home/yitai/drosophila_vision/figure_digitization/gruntman18/2b_digitized.png")
```

Expected: the image should load visually.

4. If `view_image` works, compare these two images visually:

```text
/home/yitai/drosophila_vision/figure_digitization/gruntman18/2b.png
/home/yitai/drosophila_vision/figure_digitization/gruntman18/2b_digitized.png
```

The user wants specific visual defects in `2b_digitized.png`, especially:

- which trace/color/panel has discontinuities
- which trace/color/panel suddenly spikes too high
- where the digitized line shape is visibly distorted compared with the source

Do not answer from CSV-only stats if visual loading works.

## If It Still Fails

If `view_image` still reports `bwrap: Creating new namespace failed: No space
left on device`, then the tool host is still ignoring or not inheriting
`sandbox_mode = "danger-full-access"`.

Useful checks:

```bash
pgrep -a codex
codex doctor
cat /proc/sys/user/max_net_namespaces
```

System-level fix, requiring admin/root:

```bash
sudo sysctl -w user.max_net_namespaces=1024
```

Rollback for the user-wide Codex workaround:

```bash
cp /home/yitai/.codex/config.toml.bak-20260812-sandbox /home/yitai/.codex/config.toml
```

