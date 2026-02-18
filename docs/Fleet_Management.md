# Fleet Management (Optional Queen/Worker Mode)

This mode is optional. BumbleBox can run as:

- `standalone`: normal single-box operation.
- `queen`: orchestration node for worker boxes.
- `worker`: worker node managed by a queen.

## Interface-Only Queen vs Active Queen

The queen can be either:

- interface-only (no local recording/tracking)
- active (also runs local recording/tracking)

Control this with `fleet.queen_local_pipeline_enabled`:

- `false`: interface-only queen.
- `true`: queen also runs local pipeline.

CLI setup examples:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet init-queen --queen-interface-only
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet init-queen --queen-bbox-active
```

GUI setup:

- Open `Fleet` tab.
- In `Queen Setup`, choose `--queen-interface-only` or `--queen-bbox-active`.

## Minimal Queen/Worker Setup

1. Initialize queen:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet init-queen --queen-interface-only
```

2. Enroll each worker:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet enroll-worker --host 192.168.1.21 --name worker-1
```

3. Run health checks:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet status
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet latest-status
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet discover
```

4. Optional: split latest media flow on queen:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet queen-pull-latest
python3 /Users/aec/Desktop/BumbleBox/bbx.py fleet queen-track-latest --cooldown-minutes 60
```

This keeps two distinct per-worker pointers on queen:

- `latest/latest_video.mp4` (+ `latest/latest_video.json`)
- `latest/latest_tracked.mp4` (+ `latest/latest_tracked.json`)

`latest_video` is refreshed by pull runs. `latest_tracked` is refreshed by track runs.

`fleet latest-status` adds a side-by-side per-worker view:

- latest pulled timestamp/path
- latest tracked timestamp/path
- lag in minutes
- optional online/offline status (via SSH reachability probe)

`fleet discover` scans LAN neighbors and probes reachability to flag workers that look offline.

## What `fleet status` checks

Per worker (SSH-based):

- reachability
- clock offset vs queen
- free disk on `/`
- available RAM
- CPU temperature
- system load
- BumbleBox timers visible in `systemctl list-timers`
- latest `*_run_summary.json` under worker `data_root`

## Suggested Time Sync

Use `chrony` for clock sync and SSH for orchestration:

- Queen: allow worker subnet in `/etc/chrony/chrony.conf`.
- Worker: add queen as NTP server (`server <queen_host> iburst`).

## Queen Track Guardrails

`fleet queen-track-latest` includes overload guardrails so the queen can avoid self-saturation:

- load guard: `--max-queen-load-1m` (default `3.0`)
- memory guard: `--min-queen-mem-gb` (default `0.8`)
- cycle cap: `--max-videos-total`
- repeat suppression for unchanged latest video: `--cooldown-minutes`

If queen local recording/tracking is enabled, the command blocks by default.
Use `--allow-when-queen-bbox-active` only if you intentionally want concurrent workloads.

GUI path:

- `Fleet` tab -> `Queen Latest Sync + Tracking` (Advanced view).
- `Refresh Latest/Online Matrix` updates side-by-side latest media status.
- `Discover LAN Hosts` scans the local network and warns if configured workers are unreachable.
- `Auto-Set Max Videos From Workers` sets `max_videos_total` from enabled worker count.

## Hourly Tracking Schedule

For unattended operation, set `fleet.queen_media_schedule.enabled: true`, then run:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py systemd-write
python3 /Users/aec/Desktop/BumbleBox/bbx.py systemd-install
```

When enabled on a queen, `systemd-write` generates two extra timers:

- `<unit_prefix>-queen-pull-latest.timer`
- `<unit_prefix>-queen-track-latest.timer`

If queen is interface-only (`fleet.queen_local_pipeline_enabled=false`), regular local capture timers are omitted.

Recommended starting point:

- `pull_interval_minutes`: match worker recording interval
- `track_interval_minutes`: `60` (hourly)

## Why This Is Better Than Ad-Hoc SSH

- persistent worker inventory in config
- repeatable key setup flow
- one command for fleet health snapshots
- one command for latest pulled vs latest tracked lag monitoring
- one command for LAN discovery/offline worker warnings
- GUI access for non-CLI users
- clear separation between interface-only queen and active queen
