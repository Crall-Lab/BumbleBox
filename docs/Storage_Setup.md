# Storage Setup (Auto-Mount on Boot)

BumbleBox expects recording output under `system.data_root` (default `/mnt/bumblebox/data`).

Use these commands on the Pi:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py storage status
python3 /Users/aec/Desktop/BumbleBox/bbx.py storage setup --apply-config
```

What `storage setup` does:

- detects a suitable storage partition
- writes a UUID-based `/etc/fstab` entry for stable boot mounting
- mounts the target path now
- optionally updates `system.data_root` in config (`--apply-config`)

Set a custom mount point:

```bash
python3 /Users/aec/Desktop/BumbleBox/bbx.py storage set-mount-point --mount-point /mnt/my_drive/bumblebox
python3 /Users/aec/Desktop/BumbleBox/bbx.py storage setup --mount-point /mnt/my_drive/bumblebox --apply-config
```

If storage is already mounted correctly, status shows:

`Writing data to <device name> storage, you can find it at <mount point>`

`storage status` also ends with a line saying whether a recording would be
allowed to start right now:

```text
Recording allowed: yes (the drive listed in /etc/fstab for /mnt/bumblebox/data is mounted; 212.4 GB free)
```

A recording is refused when the data drive is missing or nearly full, so data
is never written to the SD card by mistake. See `docs/Unattended_Safety.md`.

GUI path:

- Open `Doctor` tab.
- In `Storage Setup`, set mount point if needed.
- Click `Refresh Storage Status`.
- If not mounted, click `Setup Storage Auto-Mount`.
