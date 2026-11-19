# Protections for Unattended Recording

Three checks keep a box that is recording on its own from losing data quietly.
Each one fails loudly: the run exits with an error that shows up in the journal
and in `bbx systemd-status`.

## 1. Recording is refused when the data drive is missing

If the external drive is not mounted, its mount point is still an ordinary
folder on the SD card. Recording into it fills the system disk while everything
looks normal. Before a run writes anything, it now checks where the data will
go and stops if the drive is not there:

```text
Run failed to start: The data drive is not mounted: /etc/fstab lists a drive for
/mnt/bumblebox/data, but nothing is mounted there. Recording was refused so that
data is not written to the system disk under /mnt/bumblebox/data. ...
```

The rule is set by `system.require_mounted_storage`:

| Value | A mounted drive is required when |
|---|---|
| `auto` (default) | `/etc/fstab` lists a drive for `data_root` or a folder above it, or `data_root` is under `/mnt` or `/media` |
| `true` | always; `data_root` may never be on the system disk |
| `false` | never; recording to the system disk is allowed |

`bbx storage setup` writes the `/etc/fstab` entry, so a box set up that way is
protected without further changes.

The Data Drive window (Overview page, `docs/Storage_Setup.md`) mounts a drive
at the data folder and sets the value to `true`, so the box stays protected
even when the drive is not kept after a restart. When a run is refused for a
missing drive, the window that reports it offers **Mount Data Drive...**.

A run is also refused when less than `system.min_free_space_gb` (default 1 GB)
is free. Set it to `0` to turn that off.

To see what a run would decide right now:

```bash
bbx storage status
```

The last line reads `Recording allowed: yes (...)` or `Recording allowed: NO`
with the reason. The storage alert in the GUI uses the same check, so it now
reports a failure when the drive is missing.

If you record to the SD card on purpose with `data_root` under `/mnt` or
`/media`, set `system.require_mounted_storage: false`. In the Data Drive
window, **Record to the System Disk Instead...** does this and moves the data
folder to one you can write to.

## 2. Only one run uses the camera at a time

Scheduled runs can coincide. In `mixed_schedule` mode, for example, the record
and track timers both fire on the hour. A run now takes a lock on its camera
while it is capturing. A second run on that camera waits up to
`runtime.capture_lock_wait_seconds` (default 60) for the first to finish
capturing, then starts. If the wait runs out, the second run stops with:

```text
Another BumbleBox capture is using the camera (pid 4312, since 2026-11-11T14:00:00)
and did not finish within 60s, so this run was skipped.
```

The lock covers capture only. A later run can start capturing while an earlier
one is still encoding or tracking, as before. The lock is released
automatically if the process holding it exits or is killed, so it cannot be
left behind.

Runs that start in the same second also get separate session folders now. The
later one gets a numbered suffix, for example `<host>_2026-11-11_14_00_00_2`.
Before, they shared one folder and overwrote each other's files.

If runs are skipped regularly, the schedule is too tight for the recording
length; `bbx schedule-check` shows why.

On a Pi with two cameras each camera has its own lock, so the two record at
the same time while runs on one camera still take turns. See
`docs/Two_Cameras.md`.

## 3. A hung run is stopped

A scheduled run that hangs (a stalled camera, a stuck encoder) used to stay
running forever, and no later run of the same timer could start. The generated
systemd services now carry a time limit:

```ini
TimeoutStartSec=3620
```

`scheduling.run_timeout_minutes` sets it. The default, `auto`, allows the
recording time plus the larger of 30 minutes and two schedule intervals. A
20-second recording every 30 minutes gets 60 minutes and 20 seconds.

With a long interval that rule would be far too generous, so the limit is
also never later than five minutes before the next slot, whenever the
interval leaves room for the recording plus 30 minutes. A two-hour recording
once a day gets 23 hours 55 minutes; a 20-second recording every hour gets
55 minutes.

A long recording has a faster safeguard as well: a watchdog that restarts a
failed or stalled recorder within minutes. See `docs/Long_Recordings.md`.

When the limit is reached, systemd stops the run and marks it failed. The next
scheduled run starts normally. In stream mode the video recorded so far is
kept; tracking that was still in progress is lost and can be redone with
`bbx track-videos`.

Set a number of minutes if your runs legitimately take longer than the `auto`
allowance, for example long recordings followed by slow tracking.

**The limit is written into the service files.** After upgrading, or after
changing the setting, write and install the timers again:

```bash
bbx systemd-write
bbx systemd-install
```

Starting automated recording from the GUI does both.

## 4. No catch-up run after downtime

The generated timers used to be marked `Persistent=true`. With that setting,
if any slot was missed while the Pi was off, systemd started the job once,
immediately at boot (one run per timer, however long the Pi was off). A missed
recording cannot be recovered, so that run only added an off-schedule
recording while the system was still starting up.

The timers now use `Persistent=false`. After a reboot each timer waits for its
next clock-aligned slot and carries on. Slots missed while the Pi was off are
skipped.

Like the time limit, this is written into the unit files, so it reaches an
existing box only after `bbx systemd-write` and `bbx systemd-install`.

## What the GUI shows

The Overview page of `bbx gui` reports these checks without running a command.
It refreshes every 30 seconds.

| Card | What it tells you |
|---|---|
| Data storage | Whether the drive is mounted and how much space is free, or that recording will be refused and why. Once a recording exists, also the storage used per day and how many days of space are left (see below) |
| Automated recording | Whether the schedule is on, when the next recording is, and whether the last scheduled run worked. A failed run shows its reason, for example the missing drive |
| Latest recording | When the newest recording in the data folder was made and whether it succeeded |

The Automated recording card also warns when:

- the settings have changed since the schedule was started, so the installed
  timers are out of date (press **Start Automated Recording** to apply them);
- the schedule would stop when the user logs out, because lingering is off;
- the schedule is running now but would not come back after a restart.

**Start Automated Recording** checks the drive and the memory and timing
estimates from `bbx schedule-check` first. If it finds problems it lists them
and asks before starting. The estimates are approximate, so you can still
choose to start.

A command that fails in the GUI shows its reason in a message and in a red
line above the command output, instead of only an exit code.

**Cancel** appears beside that line while a command runs. It asks the command to
stop the way Ctrl+C does, so a recording writes its summary (marked "Cancelled
before it finished") and an encode-while-recording run keeps the video recorded
so far. Frames still held in memory are lost. A recording asks for confirmation
first. If a command does not stop within 15 seconds it is terminated, and after
5 more seconds killed. Cancel has not been tested with two-Pi capture, where
the other Pi may keep recording.

**Stop Automated Recording** asks before stopping, and says when the next
recording was due. A recording already in progress finishes normally; only
later ones are not started.

### Storage per day

The estimate uses the size of the newest session folders (video, tracking files
and everything else), divided by their frame count, times the frames per
recording and recordings per day the current settings would make. Only
successful recordings with the same resolution, video format, encoder speed and
colour setting count, because those decide how big a frame is. "Days of space
left" counts down to `system.min_free_space_gb`, where recording is refused.

When the video settings change, there is no matching recording yet. The
estimate then falls back to an encode benchmark run on one of your recordings
(the Settings page does this), which measures the video file alone. Otherwise
it reads "Storage per day not known yet" until one recording has been made.

The card turns amber when less than a week of space is left.

The result of the last scheduled run comes from systemd, which forgets it at
reboot. After a restart the card reads "No scheduled run has happened yet"
until the next slot.

## What these do not cover

- The live monitor and camera preview do not take the capture lock. A scheduled
  run that starts while one of them is open can still fail to open the camera.
- In two-Pi distributed capture, a node that records only thermal or depth
  (no RGB camera) does not run the storage check.
- These protections have been tested with the mock camera and simulated
  drives, not yet on a Raspberry Pi with a real drive unplugged.
- A run that is killed outright now still appears in Results, marked "Did not
  finish"; see `docs/Long_Recordings.md`.
