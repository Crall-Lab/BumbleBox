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
| Tuning file | The connected camera's NoIR file when `camera.infrared` is true, its standard file otherwise | **Infrared light** on the Settings page; `camera.tuning_file` names a file by hand |
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
| `camera.infrared` | tuning file | **Infrared light** on the Settings page: the camera's NoIR tuning file, or its standard one |
| `camera.tuning_file` | tuning file | A tuning file named by hand, used whatever `camera.infrared` says |

A new setup sends one of the first four: `HighQuality` noise reduction. A
control a camera does not offer is reported in the run's output and the
others are still applied.

### Infrared light and the tuning file

A tuning file tells the camera how to build a picture from the sensor: white
balance, colour, lens shading. Raspberry Pi supplies two for most sensors,
for example `imx477.json` and `imx477_noir.json` for the HQ camera. The NoIR
file is for a camera with no infrared filter: one sold as NoIR, or an
ordinary one with its filter taken off.

- **Infrared light** in the Picture group of the Settings page is
  `camera.infrared`. Ticked, the camera is opened with its NoIR file;
  unticked, with its standard file. A new setup has it ticked.
- **It does not depend on the camera model.** `camera.model: "hq"` with
  Infrared light ticked uses `imx477_noir.json`. Models sold with a filter
  (Camera Module 3 Standard and Wide, OwlSight) are no longer refused; the
  camera check adds a reminder that NoIR tuning suits them only with the
  filter removed.
- **The camera says which sensor it is.** When the camera is opened it is
  asked for its sensor name, and that picks the file. `camera.model` can stay
  on `auto`, and the size being captured no longer matters. Without a camera
  to ask, the model and then the recording size are used as before.
- **A camera with no NoIR file** (the OwlSight, for one) is opened with its
  standard tuning. The health check warns about this; the recording is not
  refused.
- **A tuning file the camera will not open with** does not stop a recording
  either. The camera is opened again with its standard tuning, the run's
  output says "could not be opened with tuning file", and the run summary
  shows `Resolved tuning file: default`.
- **The named camera profiles set it.** HQ Reference switches it on, OwlSight
  Reference off, and the checkbox is locked under either until **Use Custom
  Camera Settings** is pressed.
- **The live monitor has its own switch**, NoIR Tuning. It follows this
  setting until it is used, and after that stays where it was last left
  (`docs/Live_Camera_Monitor.md`).

Three faults in how the tuning file reached the camera were corrected in
November 2026:

1. **The file was probably not used at all.** libcamera chooses a camera's
   tuning file when its camera manager starts, and the manager starts the
   first time the cameras are listed. BumbleBox lists the cameras before
   opening one, and Picamera2 (0.3.39) keeps that manager, so the file named
   at opening time arrived too late. The manager is now let go before the
   camera is opened. This comes from reading the Picamera2 and libcamera
   source; it has not been seen on a Pi. An older Picamera2 that does not keep
   the manager would not have had the fault.
2. **The live monitor chose no file** when `camera.model` was `auto`, because
   it opens the camera at 1920 x 1440 and the file was guessed from the size.
3. **A Pi 4 could be handed a Pi 5 file.** Both folders (`pisp` for the Pi 5,
   `vc4` for earlier ones) are installed on a Pi, and the Pi 5 folder was
   searched first. Only the folder for the Pi in use is searched now.

If the first fault was real on a box, its recordings so far were made with
the standard tuning even with `infrared: true`, and recordings from now on
will be balanced differently. To keep the earlier look, untick Infrared light.

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
- That the NoIR tuning file takes effect: with the live monitor open, switch
  NoIR Tuning off and on, and watch the picture and the white balance numbers
  under it change. On a Pi 4 and a Pi 5.
- That recordings, the preview and the camera check still open the camera
  after the change to how it is opened.
