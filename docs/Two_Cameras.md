# Two Cameras on One Raspberry Pi

A Raspberry Pi 5 has two camera connectors. BumbleBox can record from a
Raspberry Pi camera on each, at the same moments, with one schedule or two.

This has been tested with mock cameras on a desktop. **It has not been run on
a Pi with two real cameras.** Do the load test below before relying on it.

## How it is organised

Each camera has its own config file:

| | File | Session names | Timers |
|---|---|---|---|
| Camera 1 | `config.yaml` | `<host>_<date>_<time>` | `bumblebox-v2-*` |
| Camera 2 | `config.camera2.yaml` (next to it) | `<host>_cam2_<date>_<time>` | `bumblebox-v2-cam2-*` |

Everything that works for one camera works for each of the two: recording,
tracking, the checks, the timers. Both cameras can write into the same data
folder, because Camera 2's sessions and files carry `cam2` in their names.
A day's folder then holds both cameras' sessions side by side.

Each camera has its own capture lock, so the two record at the same time.
Two runs on the *same* camera still wait for each other.

Camera 2 starts as a copy of Camera 1's recording settings with its schedule
off. Thermal, depth and two-Pi capture are switched off in it; those devices
exist once and stay with Camera 1.

## Setting it up

In the GUI: **Hardware → Add Second Camera**. It asks for Camera 2's colony ID
and data folder (the same folder as Camera 1 is the default), then shows a
**Camera 1 / Camera 2** switch at the top right of the window.

From the command line:

```bash
bbx second-camera add
```

```bash
bbx second-camera status
```

`status` lists both cameras, whether they are pinned, and whether they can run
together.

## Pinning: why it matters

libcamera numbers the cameras it finds 0, 1, ... in the order it detects them.
The number is not tied to a connector. If one camera is unplugged or fails,
the other becomes camera 0. A config that only says "camera 0" would then
record the other nest into this camera's data, with no error.

So when both cameras are connected while you add Camera 2, each config is
**pinned** to one physical camera by its libcamera Id (`camera.camera_id`).
A pinned config whose camera is missing refuses to record:

```text
Run failed to start: The camera assigned to this config is not connected
(camera.camera_id = /base/axi/pcie@120000/rp1/i2c@80000/imx477@1a). ...
Recording was refused rather than using a different camera.
```

If only one camera was connected at the time, the cameras are matched by
number and the Hardware page says so. Connect both, then press **Pin Cameras**
or run:

```bash
bbx second-camera pin
```

Swapping the two ribbon cables swaps which nest each config records. Check
each camera after any re-cabling (**Hardware → Check Camera**, with the camera
picked at the top right).

## Using the GUI with two cameras

- **Camera switch (top right):** every page shows one camera at a time. The
  page label reads, for example, `RUN · CAMERA 2`.
- **Overview:** a "Both cameras" strip shows each camera's schedule state, and
  turns red for a camera whose last run failed or whose storage is refusing.
- **Run:**
  - *Record Camera N Now* records the shown camera.
  - *Record Both Now* records both at the same moment.
  - *Start/Stop Automated Recording* act on both cameras while
    **Start and stop both cameras together** is ticked. Untick it to act on
    the shown camera only.
- **Results:** each camera lists only its own recordings.
- **Settings:** belong to the shown camera. When you change one camera's run
  mode, recording length or interval, saving asks whether the other camera
  should use the same schedule.
- **Data storage card:** when both cameras fill the same drive, the days of
  space left count both, for example
  `About 7.3 GB per day (15 GB with Camera 2) · about 14 days of space left`.

Starting the schedule also checks the two cameras together: the memory both
hold when they capture at once, and that they do not share a camera, timer
names or session names.

## Same schedule or different

Timers are aligned to the clock. Two cameras with the same interval record at
the same moments. With different intervals they still coincide whenever both
have a slot, at least at midnight, so the Pi must be able to carry both
at once either way.

The two recordings are separate processes. They start within about a second
of each other and their frames are **not** matched up; each recording has its
own frame timestamps. Hardware frame synchronisation is not implemented.

## Load: do this test on the Pi

Two cameras double the work:

- In memory mode, two sets of frames are held at once (about 18.5 MB per frame
  per camera at 4056 x 3040).
- In encode-while-recording mode, two encoders must each keep up.
- Tracking runs for both recordings in each slot.

`bbx encode-benchmark` and each camera's schedule check measure one camera
alone. To test the pair:

```bash
bbx run-once --all-cameras
```

or **Record Both Now** in the GUI. Check that both report success and that
each camera's actual frame rate (Results page) is what you asked for. Repeat
with the recording length and mode you plan to use.

## Removing the second camera

**Hardware → Remove Second Camera**, or:

```bash
bbx second-camera remove
```

This stops Camera 2's schedule, then renames its config to
`config.camera2.removed-<date>.yaml`. Its recordings are not touched. Camera 1
stays pinned, which is harmless.

## Settings reference

```yaml
camera:
  camera_num: 0        # which camera, by number (0 is the first)
  camera_id: null      # pins one physical camera; decides when set
  session_tag: ""      # added to session and file names, e.g. "cam2"
```

A one-camera box leaves all three at their defaults and opens its camera
exactly as before.

## Not covered

- Not run on a Raspberry Pi with two real cameras.
- Frames from the two cameras are not synchronised.
- `bbx export-bundle` and `bbx fps-sweep` pick the newest run in the data
  folder, which may be either camera's when they share it.
- The live monitor shows the camera picked in the GUI, or both side by side
  with its **Both Cameras** button. With both shown they share one NoIR
  tuning choice (`docs/Live_Camera_Monitor.md`).
- More than two cameras are not supported.
