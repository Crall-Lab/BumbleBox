# Stream Recording

BumbleBox can hold a recording in two ways while it is being captured. The
choice is `capture.frame_buffer` in the config.

| | `ram` (default) | `stream` |
|---|---|---|
| Where frames go | Every frame stays in memory; the video is encoded after capture ends | Frames are encoded to disk while capture continues |
| Recording length | Limited by RAM (about 18.5 MB per frame at 4056x3040) | Limited by disk space |
| Capture speed | Highest: nothing competes with the camera | Depends on the encoder keeping up |
| Tracking input | Frames in memory, or the saved video | Always the saved video |
| If the run is killed | The recording is lost | Frames encoded so far are kept |

Use `ram` for short, fast bursts. Use `stream` for anything long.

## Choosing an encoder

Stream mode only works if the encoder is at least as fast as the camera. Measure
it on the machine that will record:

```bash
bbx encode-benchmark
```

This needs no camera. It encodes test frames at `camera.width` x `camera.height`
with several encoders and prints frames per second, file size, and whether each
one keeps up with `camera.fps_target`. It finishes by printing the config
settings for the encoder it suggests.

Useful options:

- `--source-video <file>`: use frames from one of your own recordings instead
  of synthetic frames. Real nest footage gives a closer estimate.
- `--fps <n>`: test against a different frame rate.
- `--encoders mjpeg,libx264:ultrafast,libx264:veryfast`: choose what to test.
- `--frames <n>`: test frames per encoder. They are all held in memory.

The benchmark measures the encoder alone. The camera pipeline also uses CPU
during a real recording, so treat a "marginal" result as too slow.

## Turning it on

```yaml
capture:
  frame_buffer: "stream"
  stream_queue_mb: 512

camera:
  codec: "mp4"          # or "mjpeg"
  mp4_codec: "libx264"
  mp4_preset: "auto"    # "auto" means "ultrafast" in stream mode
```

To try it for one run without editing the config:

```bash
bbx run-once --frame-buffer stream --mode record_only
bbx run-once --frame-buffer stream --mp4-preset veryfast
```

## Checking that it kept up

The run summary reports what happened:

```text
Frame buffer: stream
Stream encoder: libx264 ultrafast; most frames waiting 3 (56 of 512 MB); capture waited 0.00s in 0 stall(s)
```

- **Most frames waiting** is the largest backlog during the run. A backlog that
  stays small means the encoder kept up.
- **Capture waited** is the time the camera loop spent blocked because the
  backlog reached `capture.stream_queue_mb`. Any wait lowers the achieved frame
  rate, and the run adds a warning. Fix it with a faster preset, `mjpeg`, a
  lower frame rate or resolution, or a larger `stream_queue_mb`.

Frames are never dropped silently. When the encoder cannot keep up, capture
slows down and the real capture times are still written to the frame timestamp
file.

## Checking that tracking still finds your tags

RAM recordings are tracked from the camera's raw frames. Stream recordings are
tracked from the saved video, which has been compressed. To see whether that
changes what the tracker finds on your footage, run this with tagged bees (or
a printed tag sheet) in view:

```bash
bbx tracking-compression-check
```

It captures a 4-second burst into memory, tracks those raw frames, then
encodes the same frames with each encoder and tracks the videos. Every row
holds the same frames, so differences come from the video path alone.

```text
source                         tags per frame  kept %  missed  extra   net %  shift px  MB/frame
raw frames (RAM mode)           376      37.6       -       -      -       -         -     18.50
lossless video (reference)      376      37.6    99.2       3      3    +0.0      0.00      7.48
mjpeg                           358      35.8    94.7      20      2    -4.8      0.00      2.44
libx264:ultrafast               376      37.6    98.4       6      6    +0.0      0.00      3.91
```

- **kept %** is the share of raw-frame detections also found in that video.
- **missed** were found only in the raw frames; **extra** only in the video.
  Both are mostly tags at the edge of readability flipping one way or the
  other. Extra ones may be real tags or false readings; the check cannot tell.
- **net %** is the change in total detections. A large negative value is the
  sign of real loss.
- **lossless video** has lost no pixels to compression, so its row shows what
  reading from a video costs by itself. It is the floor for every encoder.

The table above is from a synthetic scene and is only an illustration. Run the
check on your own setup, with at least a few hundred detections, before
choosing an encoder. Useful options: `--seconds <n>` for a longer burst,
`--encoders ...` to choose what to compare, and `--keep-videos` to look at the
encoded clips. Tracking CSVs and a JSON report are saved in a new folder under
`system.data_root`.

Pick an encoder using both tools: `encode-benchmark` says which ones are fast
enough, and this check says which ones keep your tags.

## How it works

1. The capture loop hands each frame to a background thread and carries on.
2. That thread feeds an `ffmpeg` process, which encodes as frames arrive.
3. The recording is written to `<session>.capture.mkv` first, because a
   Matroska file stays readable if the run is killed.
4. When capture ends, the file is copied (not re-encoded) into the final
   `.mp4` or `.avi` with the measured frame rate, and the `.mkv` is removed.
   This copy needs free space equal to the size of the recording.

If capture fails part-way, the frames recorded so far are kept in the
`.capture.mkv` file and the run summary names it.

## Things to know

- **Tracking reads the saved video** in stream mode, whatever
  `pipeline.tracking_source` says. Compressed video is not identical to the raw
  frames RAM tracking uses; run `bbx tracking-compression-check` (above) on
  your own footage before relying on it.
- **Faster presets make larger files.** `ultrafast` output is roughly twice the
  size of `veryfast`.
- **Slower presets buffer frames inside ffmpeg.** `stream_queue_mb` limits the
  frames BumbleBox holds while they wait for the encoder. The encoder also
  keeps its own look-ahead frames: none for `ultrafast` and `mjpeg`, about 10
  for `veryfast`, and 30 or more for `fast` and slower. At 4056x3040 that is
  several hundred megabytes on top of the queue for the slower presets.
- **MP4 frame rate is approximate.** The `.mp4` container rate can differ from
  the measured rate by a fraction of a percent on short clips. The exact value
  is in `<session>_actual_fps.txt` and the frame timestamp CSV, which is what
  tracking and cleaning use.
- **`track_only` runs still use RAM**, because they do not save a video.
- **`bbx fps-sweep` does not measure stream mode yet.** Use `encode-benchmark`
  and a real stream-mode run instead.
- **Not yet validated on a Raspberry Pi.** Stream mode has been tested with the
  mock camera only. Run `encode-benchmark` and a real `run-once` on the Pi
  before using it for an experiment.
