# Long Recordings

Recording for minutes or hours at a time is a different job from a 20-second
burst. A burst that fails is followed by another half an hour later. A
two-hour session that fails 15 minutes in has nothing behind it until
tomorrow. Three things close that gap.

This has been tested with a mock camera that waits between frames like a real
one, including killing and freezing the recorder mid-session. **It has not
been run on a Raspberry Pi.**

A long recording needs `capture.frame_buffer: stream` (see
`docs/Streaming_Capture.md`); frames cannot be held in memory for hours.

## 1. The recording is split into consecutive files

In stream mode a new video file is started every `capture.chunk_minutes`
(default 10), without pausing the camera. Frames keep arriving while one
encoder closes and the next opens, so no frame is lost at a boundary.

```text
<session>_part001.mp4   <session>_part001_frame_timestamps.csv
<session>_part002.mp4   <session>_part002_frame_timestamps.csv
...
<session>_frame_timestamps.csv      (the whole session, written at the end)
```

- Each file is put in its final form, with its measured frame rate and its own
  timestamps, as soon as it closes. A failure later in the session costs at
  most the file being written.
- Frame numbers run through the whole session: the first frame of part 2
  follows the last frame of part 1 in the timestamps and in tracking output.
- A recording shorter than one chunk stays a single file with its usual name.
- Tracking reads the files in order and writes one set of results.
- Set `chunk_minutes: 0` to keep one file.

Files are cut by frame count, so if the camera runs slower than its target
rate each file covers proportionally more time.

## 2. A run that is cut short is still recorded as a run

Every run writes a summary marked "Did not finish" when it starts, and
replaces it with the real one at the end. As each video file closes, that
summary is updated with the files saved so far. A run that is killed, or
loses power, therefore shows up in Results as failed, with what it kept.

## 3. A failed or stalled recording is started again

The recorder runs as a child of a small watchdog. The recorder signals
progress for every frame and every status message. The watchdog acts when:

- the recorder exits with an error before capture has finished; or
- no signal arrives for `capture.stall_seconds` (default 120). A camera that
  stops delivering frames and an encoder that stops accepting them both look
  like this.

It stops the recorder (asking first, then killing it) and starts a new
recording for the time left in the session. Each attempt is an ordinary run
with its own session folder, so afterwards Results shows a failed run
followed by a resumed one, and the gap between them is visible.

So that a camera which is truly broken cannot spin the Pi in a loop:

| Limit | Value |
|---|---|
| Pause before a restart | 5 s, then 15 s, 30 s, 60 s |
| Restarts per session | `capture.max_restarts` (default 5) |
| Too little left to record | no restart with under 30 s of the session left |

Once capture has finished the watchdog only waits. Encoding and tracking can
be slow without being stuck, and a failure there cannot be fixed by recording
again.

A session that needed a restart ends with a failure status even if the rest
was recorded, because part of it is missing. The last line of its output
says what happened:

```text
Recording failed part-way: the recorder was killed (signal 9). It was restarted
1 time(s) and the rest of the 2 h session was recorded as a separate run. ...
```

`capture.restart_on_failure` controls it:

| Value | Meaning |
|---|---|
| `auto` (default) | on for recordings of two minutes or more |
| `true` | always |
| `false` | never |

`bbx run-once --no-restart` turns it off for one run.

## The service time limit

The watchdog handles a stuck recorder within minutes. The systemd time limit
(`docs/Unattended_Safety.md`) remains as the backstop for anything else. With
`scheduling.run_timeout_minutes: auto` it is now never later than five
minutes before the next slot, when the interval leaves room for that. A
two-hour recording once a day gets 23 h 55 min. Before, the same schedule
got 50 hours, so a stuck run would have taken the next day's recording too.

## What is not covered

- **A reboot mid-session.** The session is not resumed after a power cut; the
  next recording is the next scheduled one. This follows the decision not to
  run catch-up recordings at boot.
- **The file being written when the recorder is killed outright.** It is left
  as `..._partNNN.capture.mkv`, usually playable, without its timestamps. A
  recorder that is cancelled or fails in an orderly way finishes that file
  properly.
- **Two-Pi capture** is not run under the watchdog.
- **The tracking overlay video** is made from the first file only.
- Each restart begins a new session folder; the parts are not joined.
- Nothing here has been measured on a Raspberry Pi: whether two encoders keep
  up for two hours, how the Pi behaves thermally, or how large the files are.
