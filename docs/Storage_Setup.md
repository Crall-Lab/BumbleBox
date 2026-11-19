# Data Drive Setup

BumbleBox saves recordings under `system.data_root` (default
`/mnt/bumblebox/data`). On a recording box that folder should be an external
drive, not the Pi's own SD card.

Plugging a drive into a Raspberry Pi is not enough. The desktop puts it at
`/media/<user>/<name>`, and only while someone is logged in. BumbleBox needs
the drive at its data folder, and needs it there after every restart. That is
what the Data Drive window does.

The window and the commands under it have been tested with stand-ins for the
system commands. **Mounting a real drive through them has not been tried on a
Raspberry Pi yet.**

## In the BumbleBox window

On the Overview page the Data storage card says what is true now. When it
reads "Data drive not mounted", its button reads **Mount Data Drive...**;
otherwise it reads **Data Drive...**. Both open the same window. A recording
that fails because the drive is missing offers the same button.

### When no drive is mounted

1. **Drive.** Every connected drive that could hold recordings is listed, with
   its name, size and format. The list updates by itself a few seconds after
   a drive is plugged in. Not listed: the Pi's own boot and system partitions,
   and drives that have no format yet (format a new drive as exFAT or ext4
   first).
2. **Folder.** Where the drive appears, which becomes the data folder.
   `/mnt/bumblebox/data` is the usual choice. A folder must be under `/mnt`,
   `/media` or your home folder, and may not be one that BumbleBox or the
   system lives in. If the folder already holds files on the system disk, the
   window says so: they are not deleted, but they cannot be seen while a
   drive is mounted there.
3. **After a restart.**
   - *Mount this drive here again automatically* (recommended). A line for
     the drive is added to `/etc/fstab`. If the drive is missing at startup
     the Pi still starts, and recording is refused.
   - *Do not mount it again.* The drive stays until the next restart or until
     it is removed. After that, recording is refused until a drive is mounted
     again from this window.

Then press **Mount Drive**. A drive the desktop had already put under
`/media/...` is moved to the chosen folder; close any file-manager windows
showing it first, or the move is refused.

### When a drive is mounted

- **After a restart** shows whether this drive comes back, with a button to
  change it: **Mount at Every Startup** or **Stop Mounting at Startup**.
- **Safely Remove Drive** unmounts the drive so that it can be unplugged. It
  is refused while a recording or tracking run is still using the drive.

The window will not change the drive while a recording is running.

### Recording without a drive

**Record to the System Disk Instead...** is for testing without a drive. It
moves the data folder to a folder you can write to (`~/BumbleBoxData` when
the current one is under `/mnt` or `/media`) and allows recording to the Pi's
own disk. Mounting a drive later switches the protection back on.

### Administrator rights

Mounting needs administrator rights. On Raspberry Pi OS the default user has
them without a password, and nothing is asked. Elsewhere a password dialog
appears. If the desktop cannot ask at all, the window shows the command to
type in a terminal instead.

## What gets saved

| Choice | `/etc/fstab` | `system.data_root` | `system.require_mounted_storage` |
|---|---|---|---|
| Mount, keep after a restart | line added for the drive | the chosen folder | `true` |
| Mount, do not keep | any line for the folder removed | the chosen folder | `true` |
| Mount at Every Startup | line added | unchanged | `true` |
| Stop Mounting at Startup | line removed | unchanged | `true` |
| Safely Remove Drive | unchanged | unchanged | `true` |
| Record to the system disk | unchanged | a folder on the system disk | `false` |

`require_mounted_storage: true` means a run is refused whenever no drive is
mounted at the data folder, wherever that folder is (see
`docs/Unattended_Safety.md`). With two cameras, a second camera that shares
the data folder gets the same two settings.

The `/etc/fstab` line names the drive by its UUID, so it does not matter which
USB port it is in:

```text
# >>> BumbleBox storage >>>
UUID=1234-ABCD /mnt/bumblebox/data exfat defaults,nofail,x-systemd.device-timeout=10,uid=1000,gid=1000,umask=0002 0 0
# <<< BumbleBox storage <<<
```

`nofail` is what lets the Pi start without the drive. Each change keeps a
dated copy of the previous file as `/etc/fstab.bumblebox.bak.<date>_<time>`.
Lines for other folders are left alone, so two cameras can record to two
drives.

## From a terminal

The window runs these commands. They can also be typed, from the BumbleBox
folder:

```bash
./bbx storage status
```

```bash
sudo ./bbx storage setup --mount-point /mnt/bumblebox/data --device /dev/sda1 --apply-config
```

| Command | What it does |
|---|---|
| `storage status` | What is mounted at the data folder, and whether a recording would be allowed to start |
| `storage setup` | Adds the `/etc/fstab` line and mounts the drive now |
| `storage mount` | Mounts the drive until the next restart; with `--forget`, also removes the folder's `/etc/fstab` line |
| `storage forget` | Removes the folder's `/etc/fstab` line; a mounted drive stays mounted |
| `storage unmount` | Unmounts the drive so it can be unplugged |
| `storage set-mount-point` | Changes `system.data_root` only |

`setup`, `mount`, `forget` and `unmount` need `sudo`. `--mount-point` defaults
to `system.data_root`. Without `--device` the first suitable drive found is
used, so name the drive when more than one is connected. `--apply-config` also
saves the folder as `system.data_root`.
`--dry-run` shows what would happen without changing anything.

`storage status` ends with a line saying whether a recording would be allowed
to start right now:

```text
Recording allowed: yes (the drive listed in /etc/fstab for /mnt/bumblebox/data is mounted; 212.4 GB free)
```

## Not covered

- Formatting a drive. That erases it, so it is left to the operator.
- NTFS drives depend on what the Pi has installed; exFAT and ext4 are the
  formats to use.
- After "Stop Mounting at Startup" the drive is expected to stay mounted until
  the restart. That depends on how systemd treats the removed line and has not
  been checked on a Pi.
