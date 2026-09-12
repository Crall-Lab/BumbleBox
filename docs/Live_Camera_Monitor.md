# Live Camera Monitor

The live camera monitor provides a local, non-recording view of every camera
assigned to the current Raspberry Pi. It is intended for focus, framing,
thermal-field, depth-coverage, and basic timing checks before an experiment.

## Start It

From the PyQt GUI, open the **Run** page and select
**Open Live Camera Monitor**.

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
./bbx live-monitor --seconds 30
./bbx live-monitor --mock --seconds 10
```

RGB dimensions must be positive even numbers because the preview uses YUV420.
Close the monitor before starting a recording or automated recording schedule.
The GUI enforces this for operator-started runs because one process cannot own
the same camera devices twice.

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
