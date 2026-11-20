# What Happens to a Frame

This page follows a picture from the camera to the screen and to the saved
video: what is changed along the way, what is not, and what can be adjusted.

The statements about the camera library come from reading the Picamera2
source (version 0.3.39). The level measurements were made on a desktop
computer with test frames. **None of this has been checked on a Raspberry Pi
with a real camera yet.**

## What the camera delivers

BumbleBox asks for a Picamera2 "preview" configuration in the `YUV420` format.
Each frame is a brightness image at full size, followed by two colour images
at half size in each direction.

| Property | Value | Where it comes from |
|---|---|---|
| Brightness and colour range | Full range, 0 to 255 | Picamera2's preview configuration uses the `sYCC` colour space |
| Frame width | Rounded down to a multiple of 64 (Pi 4) or 128 (Pi 5) | `align_configuration`; a 4056-wide setting records 4032 or 3968 wide |
| Noise reduction | `HighQuality` on a new setup; whatever `camera.noise_reduction` names | Set by BumbleBox. `Auto` sends nothing, which leaves Picamera2's default for this configuration, `Minimal` |
| Exposure time | Fixed at `camera.shutter_us` | Set by BumbleBox |
| Sensor gain | Automatic, unless `camera.analogue_gain` fixes it | The camera's auto-exposure adjusts it |
| White balance | Automatic, unless `camera.colour_gains` fixes it | The camera estimates it from the picture |
| Tuning file | The NoIR file when `camera.infrared` is true | `camera.tuning_file`, or chosen from the camera model |
| Colour | Switched off in the camera when recording in grayscale | `Saturation: 0` |

What the camera does with gain, white balance, the tuning file and `Saturation`
is libcamera's behaviour as documented; the Picamera2 source does not contain it.

## What BumbleBox does with it

| Output | Colour | Grayscale |
|---|---|---|
| Saved video, H.264 or H.265 | The camera's frame, unchanged, marked "full range" | The camera's brightness image, unchanged, marked "full range" |
| Saved video, MJPEG | The camera's values (JPEG is full range by nature) | The camera's brightness image |
| Saved video, MPEG-4 | Squeezed into video levels (16 to 235); nothing clipped | Same |
| Mid-recording PNG, overlay video, live monitor | Converted to RGB for display | The camera's brightness image |
| Tag detection | Brightness image, then a local contrast boost (CLAHE) | Same |

A recording is never converted to RGB on its way to the file. H.264, H.265,
MPEG-4 and MJPEG all store brightness plus half-size colour, which is what the
camera delivers, so there is nothing to gain from an RGB step.

The live monitor's **Contrast Boost** shows the grayscale image with the same
contrast boost the tag detector applies. It changes the display only.

## What changed in November 2026

Before, every picture went through a colour conversion that assumes "video
levels": brightness 16 means black and 235 means white. The camera uses the
whole 0 to 255 range, so the conversion clipped it.

| Camera brightness | Shown before | Shown now |
|---|---|---|
| 8 | 0 | 8 |
| 16 | 0 | 16 |
| 32 | 19 | 32 |
| 128 | 130 | 128 |
| 235 | 255 | 235 |
| 245 | 255 | 245 |

Everything at or below 16 was black and everything at or above 235 was white,
in the monitor, the PNG and the saved video, in colour and in grayscale.
Videos recorded in stream mode held the right numbers but were not marked
"full range", so players and OpenCV clipped them when reading. Tag detection
on a recording read back from such a file saw the clipped picture too.

Recordings made before the change still play back clipped. New ones look
flatter by comparison, because the dark and bright ends are kept.

"Full range" is a mark inside the video file. VLC, mpv, ffmpeg, browsers and
OpenCV respect it. A player that ignores it shows the picture as it looked
before the change.

## Grayscale or colour?

Grayscale drops the two half-size colour images. How much space that saves
depends on how much colour noise the camera produces, and has not been
measured on real footage. To measure it on a Pi, with a recording of the real
scene:

```bash
./bbx encode-benchmark --source-video /path/to/a/recording.mp4 --color
```

```bash
./bbx encode-benchmark --source-video /path/to/a/recording.mp4 --monochrome
```

The report gives megabytes per frame and gigabytes per hour for each.

Whether grayscale costs image quality depends on the light:

- **Infrared, NoIR camera.** The sensor's red, green and blue filters all pass
  infrared, so the colour images hold a tint and noise, not information.
  Grayscale keeps what matters.
- **Red light.** Almost all the signal is in the sensor's red pixels. The
  camera's brightness image is 30% red, 59% green and 11% blue, so it carries
  under a third of the red signal and adds the noise of the nearly dark green
  and blue pixels. Grayscale made this way is dimmer and noisier than the red
  channel alone. Record in colour under red light until this has been tried
  on a real camera.

Noise costs more space than colour does: a video encoder spends most of its
bits on detail that changes from frame to frame. That is why a new setup asks
the camera for `HighQuality` noise reduction (see below). What it does to file
size and tag detection has not been measured.

## What can be adjusted

These are applied inside the camera, so they change recordings, the live
monitor and what the tag detector sees. The first four are in the **Picture**
group of the Settings page; all can be set in the config.

| Setting | Control | Effect |
|---|---|---|
| `camera.analogue_gain` | `AnalogueGain` | `"auto"`, or a fixed sensor gain (1.0 and up). Higher is brighter and noisier. Fixed, brightness and noise stay the same from one recording to the next |
| `camera.colour_gains` | `AwbEnable` off, `ColourGains` | `"auto"`, or a fixed white balance as `[red, blue]` gains. Fixed, it cannot drift under infrared or red light |
| `camera.contrast` | `Contrast` | `1.0` is the camera's own picture. Higher makes darks darker and brights brighter, in the recording itself |
| `camera.noise_reduction` | `NoiseReductionMode` | `HighQuality` (the default for a new setup), `Fast`, `Minimal`, `Off`, or `Auto` to leave the camera's own setting (`Minimal`) |
| `camera.shutter_us` | `ExposureTime` | Exposure time in microseconds |
| `camera.monochrome_output` | `Saturation` 0 | Grayscale recording |
| `camera.digital_zoom` | `ScalerCrop` | Record part of the sensor |
| `camera.tuning_file`, `camera.infrared` | tuning file | How the camera builds the picture from the sensor |

A new setup sends one of the first four: `HighQuality` noise reduction. A
control a camera does not offer is reported in the run's output and the
others are still applied.

### Noise reduction

`HighQuality` became the default for new setups in November 2026.

- **A box set up before that is not changed.** Its saved config says
  `noise_reduction: "Auto"`, which leaves the camera's own setting. To switch
  it, choose **High quality** under Noise reduction on the Settings page.
- **A config file that does not mention `noise_reduction` at all** takes the
  new default. Configs saved by BumbleBox always mention it.
- **Camera profiles no longer set it.** The HQ Reference and OwlSight Reference
  profiles used to put `Auto` back every time the config was saved, so a
  choice made on the Settings page was lost. The choice is now kept under any
  profile.
- **Frame rate.** `HighQuality` is the mode Picamera2 itself uses for still
  photographs; for video it uses `Fast`. The camera may reach a lower frame
  rate with it, and this has not been measured on a Pi. The Overview page
  shows the frame rate the latest recording reached. If that falls below
  `camera.fps_target`, choose **Fast**.
- The mode applies wherever the camera is opened: recordings, the live
  monitor, the preview, the live tracking test and calibration snapshots.

### Choosing fixed values

1. Leave sensor gain and white balance on Automatic.
2. Open the Live Camera Monitor under the experiment's lighting. The line
   under the picture shows what the camera is using, for example
   `sensor gain 3.81 | white balance red 1.92, blue 1.45`.
3. When the numbers have settled, enter them as Fixed on the Settings page
   and save.
4. Open the monitor again to see the result. The monitor reads the saved
   settings, so save first.

Under red light, a higher red gain raises the red channel's share of the
brightness image, which is what a grayscale recording keeps. Whether that
gives a cleaner picture than recording in colour has to be tried on the
camera.

### Contrast, and why new recordings look flatter

Recordings made before the levels were corrected were stretched when shown:
about 16% more contrast, with the darkest and brightest detail cut off. New
recordings are shown as the camera made them, which looks flatter next to an
old one. There are two ways to get the stronger look back:

- **When looking.** Contrast Boost in the monitor, or the contrast control of
  a video player. Nothing is lost, and the recording stays as the camera made
  it. Tag detection adds its own contrast boost either way.
- **In the camera.** `camera.contrast` above 1.0. About 1.15 should be close to
  the old look (worked out from the old stretch, not compared on a camera).
  This is saved into the recording, and like the old stretch it
  pushes the darkest and brightest detail toward black and white; unlike the
  old stretch it is a choice, applied by the camera before the picture is
  reduced to 256 levels.

Still not settable from BumbleBox; each camera reports its own list and
ranges (`picam2.camera_controls`):

| Control | Why it could matter here |
|---|---|
| `Brightness` | Applied in the camera, so it changes the recording itself |
| `Sharpness` | Edge enhancement; too much adds halos around tags |

## Still to check on a Raspberry Pi

- That a recording plays back at the camera's levels, by recording a scene
  with deep shadow and bright highlights and comparing video and PNG.
- The frame width actually recorded for a 4056-wide setting.
- File size and tag detection in grayscale against colour.
- The frame rate reached with `HighQuality` noise reduction, the default for
  new setups, at the resolution and frame rate in use; and its file size and
  tag detection against `Fast` and `Auto`.
- Grayscale under red light.
- That fixed gain, fixed white balance, contrast and each noise reduction mode
  are accepted by the cameras in use, and what they do to file size and tag
  detection.
