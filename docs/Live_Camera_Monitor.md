# Live Camera Monitor

The live camera monitor provides a local, non-recording view of every camera
assigned to the current Raspberry Pi. It is intended for focus, framing,
thermal-field, depth-coverage, and basic timing checks before an experiment.

## Start It

From the PyQt GUI, press **Open Live Camera Monitor** at the top right of the
**Overview** page, the page the window opens on.

From a terminal:

```bash
./bbx live-monitor
```

The default OwlSight/RGB preview is `1920x1440`, and display updates are capped
at 10 FPS. On the tested OwlSight this selects the full-field 4:3 binned sensor
mode and sustains roughly 9 FPS while all cameras are active. A `1280x960`
request caused libcamera to select a cropped `1920x1080` sensor mode and is
therefore not the default. The thermal camera remains at its native configured resolution.
RealSense color and depth use the configured RealSense stream profile, but the
GUI displays at most 10 frames per second from each stream.

Useful overrides:

```bash
./bbx live-monitor --rgb-width 1920 --rgb-height 1440 --display-fps 10
./bbx live-monitor --no-realsense
./bbx live-monitor --thermal --realsense
./bbx live-monitor --no-noir
./bbx live-monitor --both-cameras
./bbx live-monitor --seconds 30
./bbx live-monitor --mock --seconds 10
```

RGB dimensions must be positive even numbers because the preview uses YUV420.
Close the monitor before starting a recording or automated recording schedule.
The GUI enforces this for operator-started runs because one process cannot own
the same camera devices twice.

## What It Shows

The monitor shows only the feeds this setup uses:

| Setup | Feeds | Layout |
|---|---|---|
| RGB camera only | 1 | The feed fills the window |
| RGB + thermal | 2 | Side by side |
| RGB + RealSense | 3 | RGB large on the left; RealSense color and depth stacked on the right |
| RGB + thermal + RealSense | 4 | 2 x 2 grid |

A sensor that is switched off, or assigned to another Pi in two-Pi capture,
gets no panel. `--thermal`, `--no-thermal`, `--realsense` and `--no-realsense`
change this for one session. On a Pi with two RGB cameras, the monitor shows
the camera that is selected in the GUI, or both (next section).

## Two Cameras Side by Side

On a Pi with a second camera set up, the header has a **Both Cameras** button
(**B** toggles it).

- **Off** shows the camera selected in the main window. **On** shows both,
  camera 1 on the left and camera 2 on the right, each under its own name
  ("Camera 1 (colony 01)").
- Switching it closes the cameras and opens them again, which takes a few
  seconds.
- The monitor opens the way it was last left. `--both-cameras` and
  `--one-camera` start it one way for a single session.
- Each camera keeps its own picture settings, and its own line under the
  picture with its tuning file, sensor gain and white balance.
- Grayscale View and Contrast Boost apply to both. Either camera can fill
  the screen: double-click it or press its **Full Screen** button.
- If one camera cannot be opened, its panel says why and the other keeps
  running.
- With a thermal or depth camera as well, the two cameras share the top row
  and the other feeds sit in a row below.

**NoIR Tuning applies to both cameras at once.** libcamera loads one tuning
choice for the whole program, so two cameras in one window cannot differ. If
the two need different tuning files (two different camera models, or a file
named by hand for one of them), both are opened with their own standard
tuning and the line under each picture says so. To see such a camera with its
own tuning file, switch Both Cameras off.

Not yet tried on a Raspberry Pi: two cameras open at once in this window, and
the frame rate the display reaches with two.

The line under the RGB camera's picture also shows the tuning file the camera
was opened with, then the sensor gain and white balance it is using. Those
are the numbers to copy when fixing them on the Settings page
(`docs/Frame_Processing.md`).

## NoIR Tuning

The monitor has its own NoIR switch, the **NoIR Tuning** button in the header
(**N** toggles it, including in full screen).

- **On** opens the camera with its NoIR tuning file, for a camera with no
  infrared filter. **Off** opens it with its standard tuning.
- The monitor opens with the switch where you last left it. Until you have
  used it, it follows **Infrared light** on the Settings page
  (`camera.infrared`).
- `--noir` and `--no-noir` start it on or off for one session, without
  changing what is remembered.
- Switching it closes the camera and opens it again, which takes a few
  seconds. A camera reads its tuning file only when it is opened.
- It changes this monitor only. When it differs from the recording setting,
  the note under the header says so.
- It is remembered separately for each config file, so two cameras on one Pi
  each keep their own. The choice is kept in
  `~/.local/share/BumbleBox/live_monitor.json`, not in the config file;
  deleting that file makes the switch follow Infrared light again.
- The monitor uses the same file a recording would, although it opens the
  camera at a smaller size.
- If the config names a tuning file by hand (`camera.tuning_file`), that file
  is used and the button is greyed out.
- If the camera has no NoIR file, or cannot be opened with it, the line under
  the picture says "standard tuning (NoIR tuning was not available for this
  camera)".

## Full Screen

Any feed can fill the screen:

- press **Full Screen** on its panel, or double-click its image;
- **Esc**, the **Exit Full Screen** button, or another double-click goes back;
- **F11** toggles full screen for the first feed, or leaves it.

In full screen the window header is hidden and only that feed is drawn. The
other cameras keep running, so going back is immediate. **Space** pauses and
resumes the display in either view.

## Grayscale

Looking in grayscale and recording in grayscale are separate, and are set in
different windows:

| Control | Where | What it changes | Saved? |
|---|---|---|---|
| **Grayscale View** | This monitor, in the header | How the camera images are drawn here | No; it is for looking only |
| **Contrast Boost** | This monitor, in the header | Brings out detail in dark and bright areas of the grayscale view | No; it is for looking only |
| **Record in Grayscale** | Main window, top right of the **Run** page | Whether this camera's recordings are saved in grayscale (`camera.monochrome_output`) | Yes, straight away |

The monitor only looks: it saves no images and changes no settings (it
remembers its own NoIR Tuning switch, in a file of its own). You can
look in grayscale and record in colour, or the other way round.

- **Grayscale View** applies to the RGB camera and the RealSense color feed.
  Thermal and depth keep their colour scale, which carries the measurement.
  **G** toggles it, including in full screen. For the RGB camera it shows the
  camera's own brightness image, untouched, which is what a grayscale
  recording keeps.
- **Contrast Boost** applies the local contrast boost (CLAHE) that the tag
  detector applies before it looks for tags, so the picture is close to what
  the detector sees. It is a grayscale view: switching it on switches
  Grayscale View on, and going back to colour switches it off. **C** toggles
  it. See `docs/Frame_Processing.md` for what happens to a frame.
- **Record in Grayscale** takes effect from the next recording, for the camera
  shown in the main window. A named camera profile fixes this setting, so
  switching it under such a profile moves the camera to the Custom profile,
  keeping its other values; a message says so when that happens. The same
  setting is the "Save grayscale video" box on the **Settings** page.

## Display Architecture

- RGB, thermal, and RealSense run in independent worker threads.
- RealSense color and depth come from one synchronized RealSense frameset.
- Each displayed stream has a one-frame latest-value buffer; old preview frames
  are replaced rather than queued.
- One slow or failed camera does not stop the other panels.
- Every panel reports displayed resolution, recent display FPS, and host-side
  frame age.
- Every complete frame is fitted with its aspect ratio preserved. Empty space
  is letterboxed; the GUI does not crop frames to fill a panel.
- Thermal is enlarged with nearest-neighbor display scaling so native pixels
  remain visually explicit. Its `160x120` view is the complete Lepton frame;
  its field of view can still be narrower because of the thermal lens.
- No monitor frames are written to disk.

The monitor is local rather than an HTTP/WebRTC server. Raspberry Pi Connect
screen sharing can display it remotely without adding another camera-encoding
pipeline. A future network-native monitor should use WebRTC or similarly
bounded low-latency encoding rather than three independent HTTP MJPEG streams.

## Recording Resolution Caveat

The monitor opens OwlSight independently at the requested preview resolution.
It does not currently run alongside `run-once`. A later integrated recording
preview can request a low-resolution Picamera2 stream beside the recording
stream, but its frame rate will still be limited by the selected sensor mode.
For example, a `9248x6944` OwlSight recording mode cannot provide a genuinely
10 FPS preview when the full-resolution sensor mode itself runs near 2.6 FPS.
