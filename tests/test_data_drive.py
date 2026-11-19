"""Mounting a data drive: the fstab handling, the choices offered, and the Data Drive window.

Nothing here mounts anything or edits the real /etc/fstab: the system commands
and the file are replaced by stand-ins.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from bumblebox_v2 import cli, data_drive, storage_manager
from bumblebox_v2.config import load_config, load_defaults, save_config, validate_config
from bumblebox_v2.storage_manager import StorageDevice, parse_lsblk_output

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from bumblebox_v2.qt_env import sanitize_current_qt_env

    sanitize_current_qt_env()
    from PyQt5.QtWidgets import QApplication

    from bumblebox_v2 import box_status, qt_gui
    from bumblebox_v2.qt_data_drive import DataDriveDialog
    from bumblebox_v2.qt_gui import BumbleBoxQtGUI
except Exception:  # pragma: no cover - optional on non-GUI test hosts
    QApplication = None

EFI = "c12a7328-f81f-11d2-ba4b-00a0c93ec93b"
DATA_FOLDER = "/mnt/bumblebox/data"
FSTAB_SYSTEM = (
    "proc            /proc           proc    defaults          0       0\n"
    "PARTUUID=aaaa-01  /boot/firmware  vfat    defaults          0       2\n"
    "PARTUUID=aaaa-02  /               ext4    defaults,noatime  0       1\n"
)
LSBLK_PI = "\n".join(
    [
        'NAME="mmcblk0" PATH="/dev/mmcblk0" TYPE="disk" FSTYPE="" UUID="" LABEL="" SIZE="59.5G" MOUNTPOINT="" RM="0" MODEL="" PKNAME="" TRAN="mmc" HOTPLUG="0" PARTTYPE=""',
        'NAME="mmcblk0p1" PATH="/dev/mmcblk0p1" TYPE="part" FSTYPE="vfat" UUID="AAAA-0001" LABEL="bootfs" SIZE="512M" MOUNTPOINT="/boot/firmware" RM="0" MODEL="" PKNAME="mmcblk0" TRAN="" HOTPLUG="0" PARTTYPE="0xc"',
        'NAME="mmcblk0p2" PATH="/dev/mmcblk0p2" TYPE="part" FSTYPE="ext4" UUID="aaaa-0002" LABEL="rootfs" SIZE="59G" MOUNTPOINT="/" RM="0" MODEL="" PKNAME="mmcblk0" TRAN="" HOTPLUG="0" PARTTYPE="0x83"',
        'NAME="sda" PATH="/dev/sda" TYPE="disk" FSTYPE="" UUID="" LABEL="" SIZE="931.5G" MOUNTPOINT="" RM="0" MODEL="Extreme SSD" PKNAME="" TRAN="usb" HOTPLUG="1" PARTTYPE=""',
        f'NAME="sda1" PATH="/dev/sda1" TYPE="part" FSTYPE="vfat" UUID="EEEE-0001" LABEL="EFI" SIZE="200M" MOUNTPOINT="" RM="0" MODEL="" PKNAME="sda" TRAN="" HOTPLUG="1" PARTTYPE="{EFI}"',
        'NAME="sda2" PATH="/dev/sda2" TYPE="part" FSTYPE="exfat" UUID="1234-ABCD" LABEL="BEES 01" SIZE="931.3G" MOUNTPOINT="/media/pi/BEES 01" RM="0" MODEL="" PKNAME="sda" TRAN="" HOTPLUG="1" PARTTYPE="ebd0"',
        'NAME="sdb" PATH="/dev/sdb" TYPE="disk" FSTYPE="ext4" UUID="9999-0000" LABEL="" SIZE="232.9G" MOUNTPOINT="" RM="1" MODEL="Cruzer Blade" PKNAME="" TRAN="usb" HOTPLUG="1" PARTTYPE=""',
        'NAME="sdc" PATH="/dev/sdc" TYPE="disk" FSTYPE="" UUID="" LABEL="" SIZE="1.8T" MOUNTPOINT="" RM="0" MODEL="New Drive" PKNAME="" TRAN="usb" HOTPLUG="1" PARTTYPE=""',
        'NAME="sdc1" PATH="/dev/sdc1" TYPE="part" FSTYPE="" UUID="" LABEL="" SIZE="1.8T" MOUNTPOINT="" RM="0" MODEL="" PKNAME="sdc" TRAN="" HOTPLUG="1" PARTTYPE="ebd0"',
        'NAME="nvme0n1p3" PATH="/dev/nvme0n1p3" TYPE="part" FSTYPE="ext4" UUID="5555-0003" LABEL="home" SIZE="400G" MOUNTPOINT="/home" RM="0" MODEL="" PKNAME="nvme0n1" TRAN="nvme" HOTPLUG="0" PARTTYPE="0fc6"',
        'NAME="nvme0n1p4" PATH="/dev/nvme0n1p4" TYPE="part" FSTYPE="swap" UUID="5555-0004" LABEL="" SIZE="8G" MOUNTPOINT="[SWAP]" RM="0" MODEL="" PKNAME="nvme0n1" TRAN="nvme" HOTPLUG="0" PARTTYPE="0657"',
    ]
)


def _devices():
    return parse_lsblk_output(LSBLK_PI)


def _device(path: str):
    return next(device for device in _devices() if device.path == path)


def _usage(free: float = 812e9, total: float = 1000e9):
    return lambda path: SimpleNamespace(free=int(free), total=int(total), used=int(total - free))


def _state(config=None, *, devices=None, fstab="", mounted=(), source=None):
    """A drive state with the system replaced: ``mounted`` lists the folders that are mount points."""
    mounts = {"/", *mounted}
    return data_drive.read_data_drive_state(
        config or load_defaults(),
        devices=_devices() if devices is None else devices,
        fstab_text=fstab,
        ismount=lambda path: path in mounts,
        mount_source=lambda path: (source, str(path)) if source else None,
        disk_usage=_usage(),
        can_write=lambda path: True,
    )


class DriveListTests(unittest.TestCase):
    def test_a_partition_is_described_by_the_disk_it_is_on(self) -> None:
        drive = _device("/dev/sda2")
        self.assertEqual(drive.model, "Extreme SSD")
        self.assertEqual(drive.transport, "usb")
        self.assertTrue(drive.hotplug)
        self.assertEqual(drive.mountpoint, "/media/pi/BEES 01")
        self.assertEqual(
            data_drive.describe_drive(drive), "Extreme SSD (USB) · 1.0 TB · exFAT · /dev/sda2"
        )

    def test_only_drives_that_could_hold_recordings_are_offered(self) -> None:
        choices = data_drive.usable_drives(_devices(), mount_folder=DATA_FOLDER)
        # Not offered: the SD card's boot and root partitions, the EFI
        # partition, the unformatted new drive, /home and swap.
        self.assertEqual([choice.path for choice in choices], ["/dev/sda2", "/dev/sdb"])
        first, second = choices
        self.assertEqual(first.title, "“BEES 01”")
        self.assertIn("now at /media/pi/BEES 01", first.detail)
        self.assertEqual(second.title, "Cruzer Blade")  # a stick formatted without a partition table
        self.assertNotIn("now at", second.detail)

    def test_sizes_are_shown_in_the_units_drives_are_sold_in(self) -> None:
        self.assertEqual(data_drive.format_size(data_drive.size_in_bytes("931.5G")), "1.0 TB")
        self.assertEqual(data_drive.format_size(data_drive.size_in_bytes("232,9G")), "250 GB")
        self.assertEqual(data_drive.format_size(data_drive.size_in_bytes("")), "size unknown")


class DriveStateTests(unittest.TestCase):
    def test_no_drive_at_a_folder_where_one_is_expected(self) -> None:
        state = _state()
        self.assertFalse(state.mounted)
        self.assertEqual(state.mount_folder, DATA_FOLDER)
        self.assertFalse(state.kept_at_startup)
        self.assertIsNone(state.preferred_choice)  # two candidates: the operator must choose
        summary = data_drive.summarize(state)
        self.assertEqual(summary.level, data_drive.LEVEL_FAIL)
        self.assertEqual(summary.headline, f"No drive is mounted at {DATA_FOLDER}")
        self.assertIn("Recording is refused", summary.detail)

    def test_a_single_candidate_or_the_startup_drive_is_preselected(self) -> None:
        only = [device for device in _devices() if device.name != "sdb"]
        self.assertEqual(_state(devices=only).preferred_choice.path, "/dev/sda2")
        fstab = FSTAB_SYSTEM + f"UUID=9999-0000 {DATA_FOLDER} ext4 defaults,nofail 0 2\n"
        state = _state(fstab=fstab)
        self.assertTrue(state.kept_at_startup)
        self.assertTrue(state.startup_drive_connected)
        self.assertEqual(state.preferred_choice.path, "/dev/sdb")
        missing = _state(fstab=FSTAB_SYSTEM + f"UUID=0000-GONE {DATA_FOLDER} ext4 defaults,nofail 0 2\n")
        self.assertFalse(missing.startup_drive_connected)
        self.assertIsNone(missing.preferred_choice)

    def test_a_drive_mounted_at_the_data_folder(self) -> None:
        devices = _devices()
        next(d for d in devices if d.path == "/dev/sda2").mountpoint = DATA_FOLDER
        fstab = FSTAB_SYSTEM + f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults,nofail 0 0\n"
        state = _state(devices=devices, fstab=fstab, mounted=[DATA_FOLDER], source="/dev/sda2")
        self.assertTrue(state.mounted)
        self.assertEqual(state.drive.path, "/dev/sda2")
        self.assertTrue(state.kept_at_startup and state.startup_is_this_drive)
        summary = data_drive.summarize(state)
        self.assertEqual(summary.level, data_drive.LEVEL_OK)
        self.assertEqual(summary.headline, f"Drive mounted at {DATA_FOLDER}")
        self.assertIn("812 GB free of 1.0 TB", summary.detail)

        other = _state(
            devices=devices,
            fstab=FSTAB_SYSTEM + f"UUID=9999-0000 {DATA_FOLDER} ext4 defaults,nofail 0 2\n",
            mounted=[DATA_FOLDER],
            source="/dev/sda2",
        )
        self.assertTrue(other.kept_at_startup)
        self.assertFalse(other.startup_is_this_drive)

    def test_a_data_folder_inside_a_drive_the_desktop_mounted(self) -> None:
        config = load_defaults()
        config["system"]["data_root"] = "/media/pi/BEES 01/bumblebox"
        state = _state(config, mounted=["/media/pi/BEES 01"], source="/dev/sda2")
        self.assertTrue(state.mounted)
        self.assertEqual(state.mount_folder, "/media/pi/BEES 01")
        self.assertEqual(state.drive.path, "/dev/sda2")
        self.assertFalse(state.kept_at_startup)
        self.assertIn("Recordings go to /media/pi/BEES 01/bumblebox", data_drive.summarize(state).detail)

    def test_a_separate_home_partition_is_not_a_data_drive(self) -> None:
        config = load_defaults()
        config["system"]["data_root"] = "/home/pi/BumbleBoxData"
        state = _state(config, mounted=["/home"])
        self.assertFalse(state.mounted)
        summary = data_drive.summarize(state)
        self.assertEqual(summary.level, data_drive.LEVEL_WARN)
        self.assertIn("system disk", summary.detail)

    def test_a_drive_that_cannot_be_written_to_is_a_failure(self) -> None:
        state = _state(mounted=[DATA_FOLDER], source="/dev/sda2")
        state.writable = False
        summary = data_drive.summarize(state)
        self.assertEqual(summary.level, data_drive.LEVEL_FAIL)
        self.assertIn("not writable", summary.headline)


class FolderChoiceTests(unittest.TestCase):
    def problem(self, folder: str, **kwargs):
        return data_drive.mount_folder_problem(
            folder,
            home=Path("/home/pi"),
            protected=[Path("/home/pi/BumbleBox"), Path("/home/pi/.config/bumblebox")],
            ismount=kwargs.get("ismount", lambda path: False),
        )

    def test_sensible_folders_are_accepted(self) -> None:
        for folder in ("/mnt/bumblebox/data", "/mnt", "/media/pi/data", "/home/pi/BumbleBoxData", "/mnt/my data"):
            self.assertIsNone(self.problem(folder), folder)

    def test_folders_that_would_hide_the_system_or_bumblebox_are_refused(self) -> None:
        self.assertIn("full path", self.problem("data"))
        self.assertIn("Enter the folder", self.problem("  "))
        self.assertIn("would hide /home/pi", self.problem("/"))
        self.assertIn("would hide /home/pi", self.problem("/home/pi"))
        self.assertIn("would hide /home/pi/BumbleBox", self.problem("/home/pi/BumbleBox"))
        self.assertIn("would hide /home/pi/.config/bumblebox", self.problem("/home/pi/.config"))
        self.assertIn("BumbleBox itself uses", self.problem("/home/pi/BumbleBox/data"))
        self.assertIn("under /mnt, /media or your home folder", self.problem("/usr/share/bees"))
        self.assertIn("plugged-in drives", self.problem("/media/pi"))
        self.assertIn("may not contain", self.problem("/mnt/a#b"))
        self.assertIn(
            "Another drive is already mounted", self.problem("/mnt/x", ismount=lambda path: path == "/mnt/x")
        )

    def test_files_already_in_the_folder_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(data_drive.items_hidden_by_mount(directory), 0)
            (Path(directory) / "session_a").mkdir()
            (Path(directory) / "notes.txt").write_text("x")
            self.assertEqual(data_drive.items_hidden_by_mount(directory), 2)
            self.assertEqual(data_drive.items_hidden_by_mount(directory, ismount=lambda path: True), 0)
            self.assertEqual(data_drive.items_hidden_by_mount(str(Path(directory) / "missing")), 0)

    def test_folder_for_recording_to_the_system_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            # /mnt/... is neither writable nor a sensible name for data on the system disk.
            self.assertEqual(data_drive.own_disk_folder(load_defaults(), home=home), str(home / "BumbleBoxData"))
            mine = {"system": {"data_root": str(home / "experiment1")}}
            self.assertEqual(data_drive.own_disk_folder(mine, home=home), str(home / "experiment1"))


class ElevationTests(unittest.TestCase):
    @staticmethod
    def which(*found):
        return lambda name: f"/usr/bin/{name}" if name in found else None

    def test_passwordless_sudo_is_preferred(self) -> None:
        elevation = data_drive.find_elevation(
            which=self.which("sudo", "pkexec"), run=lambda *a, **k: SimpleNamespace(returncode=0), euid=1000
        )
        self.assertEqual((elevation.method, elevation.prefix), ("sudo", ["/usr/bin/sudo", "-n"]))
        self.assertFalse(elevation.asks_for_password)

    def test_pkexec_asks_when_sudo_needs_a_password(self) -> None:
        elevation = data_drive.find_elevation(
            which=self.which("sudo", "pkexec"), run=lambda *a, **k: SimpleNamespace(returncode=1), euid=1000
        )
        self.assertEqual((elevation.method, elevation.prefix), ("pkexec", ["/usr/bin/pkexec"]))
        self.assertTrue(elevation.asks_for_password)

    def test_root_needs_nothing_and_no_tool_means_no_way(self) -> None:
        self.assertEqual(data_drive.find_elevation(which=self.which(), euid=0).method, "root")
        none = data_drive.find_elevation(
            which=self.which("sudo"), run=lambda *a, **k: SimpleNamespace(returncode=1), euid=1000
        )
        self.assertFalse(none.available)

    def test_commands_for_each_action(self) -> None:
        sudo = data_drive.Elevation("sudo", ["/usr/bin/sudo", "-n"])
        common = dict(config_path="/home/pi/c.yaml", mount_folder=DATA_FOLDER)
        keep = data_drive.storage_command(
            data_drive.ACTION_MOUNT_KEEP, sudo, device_path="/dev/sda2", python="/venv/python", **common
        )
        self.assertEqual(keep[:4], ["/usr/bin/sudo", "-n", "/venv/python", "-B"])
        self.assertTrue(keep[4].endswith("bbx.py"))
        self.assertEqual(
            keep[5:],
            ["storage", "setup", "--config", "/home/pi/c.yaml", "--mount-point", DATA_FOLDER, "--device", "/dev/sda2"],
        )
        once = data_drive.storage_arguments(data_drive.ACTION_MOUNT_ONCE, device_path="/dev/sda2", **common)
        self.assertEqual(once[:3], ["storage", "mount", "--forget"])
        self.assertEqual(
            data_drive.storage_arguments(data_drive.ACTION_UNMOUNT, device_path="/dev/sda2", **common),
            ["storage", "unmount", "--config", "/home/pi/c.yaml", "--mount-point", DATA_FOLDER],
        )
        self.assertEqual(data_drive.storage_arguments(data_drive.ACTION_FORGET, **common)[:2], ["storage", "forget"])
        typed = data_drive.terminal_command(
            data_drive.ACTION_MOUNT_KEEP, config_path="/home/pi/c.yaml", mount_folder="/mnt/my data", python="python3"
        )
        self.assertTrue(typed.startswith("sudo python3 -B "))
        self.assertIn("--mount-point '/mnt/my data'", typed)

    def test_every_action_is_a_command_the_cli_accepts(self) -> None:
        parser = cli.build_parser()
        for action in (
            data_drive.ACTION_MOUNT_KEEP,
            data_drive.ACTION_MOUNT_ONCE,
            data_drive.ACTION_KEEP,
            data_drive.ACTION_FORGET,
            data_drive.ACTION_UNMOUNT,
        ):
            arguments = data_drive.storage_arguments(
                action, config_path="/home/pi/c.yaml", mount_folder=DATA_FOLDER, device_path="/dev/sda2"
            )
            parsed = parser.parse_args(arguments)
            self.assertEqual(parsed.mount_point, DATA_FOLDER)

    def test_a_refused_password_prompt_is_explained(self) -> None:
        pkexec = data_drive.Elevation("pkexec", ["/usr/bin/pkexec"])
        self.assertIn("password was not given", data_drive.explain_elevation_failure(pkexec, 126))
        self.assertIn("could not ask", data_drive.explain_elevation_failure(pkexec, 127))
        self.assertIsNone(data_drive.explain_elevation_failure(pkexec, 1))
        self.assertIsNone(data_drive.explain_elevation_failure(data_drive.Elevation("sudo", ["sudo", "-n"]), 126))


class SavedChoiceTests(unittest.TestCase):
    def test_the_choice_is_written_to_each_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "config.yaml", Path(directory) / "config.camera2.yaml"
            for path in (first, second):
                save_config(path, load_defaults())
            data_drive.save_drive_choice([first, second], require_drive=True, data_folder="/mnt/bees")
            for path in (first, second):
                config = load_config(path)
                validate_config(config)
                self.assertEqual(config["system"]["data_root"], "/mnt/bees")
                self.assertIs(config["system"]["require_mounted_storage"], True)
            data_drive.save_drive_choice([first], require_drive=False)
            self.assertIs(load_config(first)["system"]["require_mounted_storage"], False)
            self.assertEqual(load_config(first)["system"]["data_root"], "/mnt/bees")

    def test_a_running_capture_is_noticed(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "capture.lock"
            with mock.patch.dict(os.environ, {"BUMBLEBOX_CAPTURE_LOCK": str(lock)}):
                config = load_defaults()
                self.assertFalse(data_drive.capture_in_progress([config]))
                lock.write_text("pid 1\n")
                self.assertFalse(data_drive.capture_in_progress([config]))
                holder = os.open(str(lock), os.O_RDWR)
                try:
                    fcntl.flock(holder, fcntl.LOCK_EX)
                    self.assertTrue(data_drive.capture_in_progress([config]))
                finally:
                    os.close(holder)
                self.assertFalse(data_drive.capture_in_progress([config]))


class FstabTests(unittest.TestCase):
    """The text that ends up in /etc/fstab. A bad line there can stop a Pi from starting."""

    def entry(self, device_path: str, folder: str) -> str:
        return storage_manager.build_fstab_entry(_device(device_path), folder, uid=1000, gid=1000)

    def test_the_entry_does_not_hold_up_startup_when_the_drive_is_missing(self) -> None:
        entry = self.entry("/dev/sda2", DATA_FOLDER)
        self.assertEqual(
            entry,
            f"UUID=1234-ABCD {DATA_FOLDER} exfat "
            "defaults,nofail,x-systemd.device-timeout=10,uid=1000,gid=1000,umask=0002 0 0",
        )
        self.assertIn("nofail", self.entry("/dev/sdb", DATA_FOLDER))

    def test_a_space_in_the_folder_name_stays_one_field(self) -> None:
        entry = self.entry("/dev/sda2", "/mnt/my data")
        self.assertEqual(len(entry.split()), 6)
        self.assertIn("/mnt/my\\040data", entry)
        self.assertEqual(storage_manager._fstab_entry_for_mount("/mnt/my data", entry + "\n"), entry)
        self.assertEqual(storage_manager.fstab_source_uuid(entry), "1234-ABCD")

    def test_a_folder_name_that_would_corrupt_the_file_is_refused(self) -> None:
        for folder in ("/mnt/a\nUUID=x / ext4 defaults 0 0", "/mnt/a#b", "mnt/relative"):
            with self.assertRaises(ValueError):
                self.entry("/dev/sda2", folder)

    def test_setting_a_drive_keeps_every_other_line(self) -> None:
        text, changed = storage_manager._update_fstab_text(
            FSTAB_SYSTEM,
            mount_point=DATA_FOLDER,
            device_uuid="1234-ABCD",
            new_entry=self.entry("/dev/sda2", DATA_FOLDER),
        )
        self.assertTrue(changed)
        self.assertTrue(text.startswith(FSTAB_SYSTEM))
        self.assertEqual(
            text[len(FSTAB_SYSTEM):].strip().splitlines(),
            ["# >>> BumbleBox storage >>>", self.entry("/dev/sda2", DATA_FOLDER), "# <<< BumbleBox storage <<<"],
        )
        again, changed_again = storage_manager._update_fstab_text(
            text, mount_point=DATA_FOLDER, device_uuid="1234-ABCD", new_entry=self.entry("/dev/sda2", DATA_FOLDER)
        )
        self.assertEqual(again, text)
        self.assertFalse(changed_again)

    def test_a_second_drive_at_another_folder_does_not_remove_the_first(self) -> None:
        first = self.entry("/dev/sda2", DATA_FOLDER)
        second = self.entry("/dev/sdb", "/mnt/bumblebox/camera2")
        text, _ = storage_manager._update_fstab_text(
            FSTAB_SYSTEM, mount_point=DATA_FOLDER, device_uuid="1234-ABCD", new_entry=first
        )
        text, _ = storage_manager._update_fstab_text(
            text, mount_point="/mnt/bumblebox/camera2", device_uuid="9999-0000", new_entry=second
        )
        self.assertIn(first, text)
        self.assertIn(second, text)
        self.assertEqual(text.count("# >>> BumbleBox storage >>>"), 1)

    def test_a_replacement_drive_takes_over_the_folder(self) -> None:
        old = f"UUID=0000-OLD {DATA_FOLDER} ext4 defaults,nofail 0 2"
        text, _ = storage_manager._update_fstab_text(
            FSTAB_SYSTEM + old + "\n",
            mount_point=DATA_FOLDER,
            device_uuid="1234-ABCD",
            new_entry=self.entry("/dev/sda2", DATA_FOLDER),
        )
        self.assertNotIn("0000-OLD", text)
        self.assertEqual(text.count(DATA_FOLDER), 1)

    def test_forgetting_a_folder_removes_only_its_line(self) -> None:
        first = self.entry("/dev/sda2", DATA_FOLDER)
        second = self.entry("/dev/sdb", "/mnt/bumblebox/camera2")
        text = FSTAB_SYSTEM + f"\n# >>> BumbleBox storage >>>\n{first}\n{second}\n# <<< BumbleBox storage <<<\n"
        without_first, removed = storage_manager._rewrite_fstab_text(text, mount_point=DATA_FOLDER + "/")
        self.assertEqual(removed, [first])
        self.assertIn(second, without_first)
        bare, removed = storage_manager._rewrite_fstab_text(without_first, mount_point="/mnt/bumblebox/camera2")
        self.assertEqual(removed, [second])
        self.assertEqual(bare, FSTAB_SYSTEM)  # no empty BumbleBox block is left behind


@contextmanager
def fake_system(fstab_text: str = FSTAB_SYSTEM, *, root: bool = True, mounted_at: str | None = None, fail=None):
    """Replace /etc/fstab and the system commands; yields the temp fstab path and the commands run."""
    with tempfile.TemporaryDirectory() as directory:
        fstab = Path(directory) / "fstab"
        fstab.write_text(fstab_text)
        commands: list[list[str]] = []
        state = {"mounted_at": mounted_at}

        def run(args, timeout=None):
            commands.append(list(args))
            if fail and args[0] == fail[0]:
                return SimpleNamespace(returncode=32, stdout="", stderr=fail[1])
            if args[0] == "mount":
                state["mounted_at"] = args[-1]
            if args[0] == "umount":
                state["mounted_at"] = None
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def findmnt(path):
            if state["mounted_at"] and str(path) == state["mounted_at"]:
                return "/dev/sda2", str(path)
            return "/dev/mmcblk0p2", "/"

        with mock.patch.object(storage_manager, "_FSTAB_PATH", fstab), mock.patch.object(
            storage_manager, "_run_command", side_effect=run
        ), mock.patch.object(storage_manager, "_findmnt_target", side_effect=findmnt), mock.patch.object(
            storage_manager, "_is_root_user", return_value=root
        ), mock.patch.object(
            storage_manager, "discover_storage_devices", side_effect=_devices
        ), mock.patch.object(
            storage_manager, "_ensure_mount_root_owned_by_user"
        ), mock.patch.object(
            Path, "mkdir"
        ):
            yield fstab, commands


class StorageOperationTests(unittest.TestCase):
    def test_setup_tells_systemd_before_mounting(self) -> None:
        with fake_system() as (fstab, commands):
            result = storage_manager.setup_storage_auto_mount(DATA_FOLDER, device_path="/dev/sda2", uid=1000, gid=1000)
            self.assertTrue(result.fstab_updated and result.mounted_now)
            self.assertIn(f"UUID=1234-ABCD {DATA_FOLDER} exfat", fstab.read_text())
            self.assertTrue(Path(result.fstab_backup_path).read_text() == FSTAB_SYSTEM)
            # systemd must know the new line before the mount, or it may undo the mount.
            self.assertEqual(
                [command[:2] for command in commands],
                [["systemctl", "daemon-reload"], ["umount", "/media/pi/BEES 01"], ["mount", DATA_FOLDER]],
            )

    def test_setup_changes_nothing_without_root_or_on_a_dry_run(self) -> None:
        with fake_system(root=False) as (fstab, commands):
            with self.assertRaises(PermissionError):
                storage_manager.setup_storage_auto_mount(DATA_FOLDER, device_path="/dev/sda2")
            result = storage_manager.setup_storage_auto_mount(DATA_FOLDER, device_path="/dev/sda2", dry_run=True)
            self.assertFalse(result.fstab_updated)
            self.assertEqual(fstab.read_text(), FSTAB_SYSTEM)
            self.assertEqual(commands, [])

    def test_mounting_once_first_drops_the_startup_entry(self) -> None:
        old = FSTAB_SYSTEM + f"UUID=0000-OLD {DATA_FOLDER} ext4 defaults,nofail 0 2\n"
        with fake_system(old) as (fstab, commands):
            result = storage_manager.mount_storage_device_now(
                DATA_FOLDER, device_path="/dev/sda2", uid=1000, gid=1000, forget_startup=True
            )
            self.assertTrue(result.mounted_now)
            self.assertEqual(fstab.read_text(), FSTAB_SYSTEM)
            self.assertEqual(commands[0], ["systemctl", "daemon-reload"])
            self.assertEqual(commands[-1][0], "mount")
            self.assertEqual(commands[-1][-2:], ["/dev/disk/by-uuid/1234-ABCD", DATA_FOLDER])

    def test_mounting_once_leaves_fstab_alone_unless_asked(self) -> None:
        old = FSTAB_SYSTEM + f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults,nofail 0 0\n"
        with fake_system(old) as (fstab, _commands):
            storage_manager.mount_storage_device_now(DATA_FOLDER, device_path="/dev/sda2", uid=1000, gid=1000)
            self.assertEqual(fstab.read_text(), old)

    def test_forgetting_removes_the_startup_entry_and_keeps_a_copy(self) -> None:
        entry = f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults,nofail 0 0"
        with fake_system(FSTAB_SYSTEM + entry + "\n") as (fstab, commands):
            result = storage_manager.remove_storage_auto_mount(DATA_FOLDER)
            self.assertTrue(result.fstab_updated)
            self.assertEqual(result.removed_entries, [entry])
            self.assertEqual(fstab.read_text(), FSTAB_SYSTEM)
            self.assertIn(entry, Path(result.fstab_backup_path).read_text())
            self.assertEqual(commands, [["systemctl", "daemon-reload"]])
            nothing = storage_manager.remove_storage_auto_mount(DATA_FOLDER)
            self.assertFalse(nothing.fstab_updated)
            self.assertIn("no entry", nothing.message)

    def test_forgetting_needs_root(self) -> None:
        entry = f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults,nofail 0 0\n"
        with fake_system(FSTAB_SYSTEM + entry, root=False) as (fstab, _commands):
            with self.assertRaises(PermissionError):
                storage_manager.remove_storage_auto_mount(DATA_FOLDER)
            self.assertEqual(fstab.read_text(), FSTAB_SYSTEM + entry)

    def test_unmounting(self) -> None:
        with fake_system(mounted_at=DATA_FOLDER) as (_fstab, commands):
            result = storage_manager.unmount_storage(DATA_FOLDER)
            self.assertTrue(result.unmounted)
            self.assertIn("can be unplugged", result.message)
            self.assertEqual(commands, [["umount", DATA_FOLDER]])
            again = storage_manager.unmount_storage(DATA_FOLDER)
            self.assertFalse(again.was_mounted)
            self.assertEqual(len(commands), 1)

    def test_a_drive_in_use_is_not_unmounted_and_says_why(self) -> None:
        with fake_system(mounted_at=DATA_FOLDER, fail=("umount", "umount: /mnt/bumblebox/data: target is busy.")):
            with self.assertRaises(RuntimeError) as raised:
                storage_manager.unmount_storage(DATA_FOLDER)
            self.assertIn("is in use", str(raised.exception))
            self.assertIn("Wait for any recording", str(raised.exception))


class StorageCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            save_config(config_path, load_defaults())
            args = cli.build_parser().parse_args([*arguments, "--config", str(config_path)])
            output = io.StringIO()
            with redirect_stdout(output):
                code = args.func(args)
            return code, output.getvalue()

    def status(self, **changes):
        base = dict(mounted=True, writable=True, fstab_entry_present=False, fstab_entry=None,
                    device_uuid="1234-ABCD", device_path="/dev/sda2")
        base.update(changes)
        return SimpleNamespace(**base)

    def test_setup_still_sets_startup_for_a_drive_that_is_already_mounted(self) -> None:
        done = SimpleNamespace(mounted_now=True)
        with mock.patch.object(cli, "get_storage_status", return_value=self.status()), mock.patch.object(
            cli, "setup_storage_auto_mount", return_value=done
        ) as setup, mock.patch.object(cli, "format_storage_setup_result", return_value="set up"):
            code, output = self.run_cli("storage", "setup", "--mount-point", DATA_FOLDER)
        self.assertEqual((code, output.strip()), (0, "set up"))
        setup.assert_called_once()

    def test_setup_has_nothing_to_do_when_startup_already_names_the_mounted_drive(self) -> None:
        ready = self.status(fstab_entry_present=True, fstab_entry=f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults 0 0")
        with mock.patch.object(cli, "get_storage_status", return_value=ready), mock.patch.object(
            cli, "setup_storage_auto_mount"
        ) as setup, mock.patch.object(cli, "format_storage_status_report", return_value="status"):
            code, _output = self.run_cli("storage", "setup", "--mount-point", DATA_FOLDER)
        self.assertEqual(code, 0)
        setup.assert_not_called()

    def test_unmount_and_forget_explain_how_to_get_root(self) -> None:
        with mock.patch.object(cli, "unmount_storage", side_effect=PermissionError("needs root")):
            code, output = self.run_cli("storage", "unmount", "--mount-point", DATA_FOLDER)
        self.assertEqual(code, 1)
        self.assertIn("sudo python3", output)
        self.assertIn("storage unmount", output)
        with mock.patch.object(cli, "remove_storage_auto_mount", side_effect=PermissionError("needs root")):
            code, output = self.run_cli("storage", "forget")
        self.assertEqual(code, 1)
        self.assertIn(f"storage forget --config", output)

    def test_a_drive_in_use_fails_the_unmount_command(self) -> None:
        with mock.patch.object(cli, "unmount_storage", side_effect=RuntimeError("The drive is in use")):
            code, output = self.run_cli("storage", "unmount")
        self.assertEqual(code, 1)
        self.assertIn("Storage unmount failed: The drive is in use", output)


@unittest.skipIf(QApplication is None, "PyQt5 is not available")
class DataDriveDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.devices = _devices()
        self.fstab = FSTAB_SYSTEM
        self.mounted: set[str] = set()
        self.choices_saved: list[tuple] = []
        self.launched: list[list[str]] = []
        self.busy: str | None = None
        self.config = load_defaults()
        self.elevation = data_drive.Elevation("sudo", ["/usr/bin/sudo", "-n"])

    def read(self, config):
        return _state(
            config,
            devices=self.devices,
            fstab=self.fstab,
            mounted=self.mounted,
            source="/dev/sda2" if self.mounted else None,
        )

    def choose(self, folder, require):
        self.choices_saved.append((folder, require))
        if folder:
            self.config["system"]["data_root"] = folder
        self.config["system"]["require_mounted_storage"] = require
        return self.config

    @contextmanager
    def dialog(self, *, confirm: bool = True):
        dialog = DataDriveDialog(
            None,
            config_path=Path("/home/pi/.config/bumblebox/config.yaml"),
            config=self.config,
            on_choice=self.choose,
            busy_reason=lambda: self.busy,
            read_state=self.read,
            elevation_finder=lambda: self.elevation,
        )
        dialog._poll.stop()
        dialog._launch = self.launched.append
        dialog._confirm = lambda *args: confirm
        try:
            yield dialog
        finally:
            dialog._working = False
            dialog.close()

    def select(self, dialog, device_path: str) -> None:
        for button in dialog.drive_group.buttons():
            if button.property("devicePath") == device_path:
                button.setChecked(True)
                return
        self.fail(f"{device_path} is not offered")

    def test_with_two_drives_the_operator_must_choose(self) -> None:
        with self.dialog() as dialog:
            self.assertFalse(dialog.no_drive_page.isHidden())
            self.assertTrue(dialog.mounted_page.isHidden())
            self.assertEqual(len(dialog.drive_group.buttons()), 2)
            self.assertIsNone(dialog.drive_group.checkedButton())
            self.assertFalse(dialog.mount_button.isEnabled())
            self.assertIn("Choose the drive", dialog.drive_note.text())
            self.assertEqual(dialog.folder_edit.text(), DATA_FOLDER)
            self.select(dialog, "/dev/sda2")
            self.assertTrue(dialog.mount_button.isEnabled())
            self.assertIn("now at /media/pi/BEES 01 and will be moved", dialog.drive_note.text())

    def test_no_drive_connected(self) -> None:
        self.devices = [device for device in self.devices if not device.name.startswith(("sda", "sdb"))]
        with self.dialog() as dialog:
            self.assertEqual(dialog.drive_group.buttons(), [])
            self.assertIn("No usable drive is connected", dialog.drive_note.text())
            self.assertFalse(dialog.mount_button.isEnabled())
            # Plugging one in is picked up by the next look.
            self.devices = _devices()
            dialog._poll_drives() if dialog.isVisible() else dialog.refresh()
            self.assertEqual(len(dialog.drive_group.buttons()), 2)

    def test_mount_and_keep_after_a_restart(self) -> None:
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sda2")
            self.assertTrue(dialog.keep_radio.isChecked())  # the recommended choice is the default
            dialog.mount_button.click()
            self.assertEqual(len(self.launched), 1)
            command = self.launched[0]
            self.assertEqual(command[:2], ["/usr/bin/sudo", "-n"])
            self.assertEqual(command[5:7], ["storage", "setup"])
            self.assertEqual(command[-4:], ["--mount-point", DATA_FOLDER, "--device", "/dev/sda2"])
            self.assertFalse(dialog.mount_button.isEnabled())  # nothing else can be started meanwhile
            self.assertFalse(dialog.close_button.isEnabled())

            self.mounted = {DATA_FOLDER}
            self.fstab = FSTAB_SYSTEM + f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults,nofail 0 0\n"
            dialog._action_finished(0, "Storage Setup\nMounted now: True\n")
            self.assertEqual(self.choices_saved, [(DATA_FOLDER, True)])
            self.assertFalse(dialog.mounted_page.isHidden())
            self.assertTrue(dialog.mount_button.isHidden())
            self.assertIn("mounted here again after every restart", dialog.result_label.text())
            self.assertEqual(dialog.result_label.objectName(), "StatusGood")
            self.assertEqual(dialog.startup_button.text(), "Stop Mounting at Startup")
            self.assertEqual(dialog.summary_card.property("tone"), "mint")
            self.assertTrue(dialog.close_button.isEnabled())

    def test_mount_for_this_session_only(self) -> None:
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sdb")
            dialog.once_radio.setChecked(True)
            self.assertIn("recording is refused until one is mounted again", dialog.restart_note.text())
            dialog.folder_edit.setText("/mnt/bees")
            dialog.mount_button.click()
            self.assertEqual(self.launched[0][5:8], ["storage", "mount", "--forget"])
            self.assertEqual(self.launched[0][-4:], ["--mount-point", "/mnt/bees", "--device", "/dev/sdb"])
            self.mounted = {"/mnt/bees"}
            dialog._action_finished(0, "Storage Mount\n")
            # The new folder becomes the data folder, and a drive is now required there.
            self.assertEqual(self.choices_saved, [("/mnt/bees", True)])
            self.assertIn("will not be mounted again after a restart", dialog.result_label.text())
            self.assertEqual(dialog.startup_button.text(), "Mount at Every Startup")

    def test_a_failed_mount_changes_no_settings_and_says_why(self) -> None:
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sda2")
            dialog.mount_button.click()
            dialog._action_finished(1, "Storage setup failed: umount: /media/pi/BEES 01: target is busy.\n")
            self.assertEqual(self.choices_saved, [])
            self.assertEqual(dialog.result_label.objectName(), "StatusFail")
            self.assertIn("target is busy", dialog.result_label.text())
            self.assertFalse(dialog.no_drive_page.isHidden())
            self.assertTrue(dialog.mount_button.isEnabled())  # ready to try again
            self.assertFalse(dialog.details_button.isHidden())

    def test_a_command_that_succeeds_without_mounting_is_a_failure(self) -> None:
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sda2")
            dialog.mount_button.click()
            dialog._action_finished(0, "done\n")  # ...but nothing is mounted
            self.assertEqual(self.choices_saved, [])
            self.assertIn("no drive is mounted", dialog.result_label.text())

    def test_a_refused_password_shows_the_command_to_type(self) -> None:
        self.elevation = data_drive.Elevation("pkexec", ["/usr/bin/pkexec"])
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sda2")
            dialog.mount_button.click()
            self.assertEqual(self.launched[0][0], "/usr/bin/pkexec")
            self.assertIn("Enter the administrator password", dialog.result_label.text())
            dialog._action_finished(126, "Error executing command as another user: Not authorized\n")
            text = dialog.result_label.text()
            self.assertIn("password was not given, so nothing was changed", text)
            self.assertIn("storage setup", text)
            self.assertIn("sudo ", text)
            self.assertEqual(self.choices_saved, [])

    def test_without_a_way_to_become_root_nothing_is_started(self) -> None:
        self.elevation = data_drive.Elevation("none")
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sda2")
            dialog.mount_button.click()
            self.assertEqual(self.launched, [])
            self.assertIn("run this in a terminal", dialog.result_label.text())
            self.assertTrue(dialog.mount_button.isEnabled())

    def test_nothing_is_changed_while_a_recording_is_running(self) -> None:
        self.busy = "A scheduled recording is running. Wait for it to finish before changing the data drive."
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sda2")
            dialog.mount_button.click()
            self.assertEqual(self.launched, [])
            self.assertEqual(dialog.result_label.text(), self.busy)

    def test_an_unsafe_folder_blocks_mounting(self) -> None:
        with self.dialog() as dialog:
            self.select(dialog, "/dev/sda2")
            dialog.folder_edit.setText("/usr/share/bees")
            self.assertFalse(dialog.mount_button.isEnabled())
            self.assertFalse(dialog.folder_note.isHidden())
            self.assertIn("under /mnt, /media or your home folder", dialog.folder_note.text())
            dialog.folder_edit.setText("/mnt/bees")
            self.assertTrue(dialog.mount_button.isEnabled())
            self.assertTrue(dialog.folder_note.isHidden())

    def test_files_already_in_the_folder_are_pointed_out(self) -> None:
        with self.dialog() as dialog, mock.patch(
            "bumblebox_v2.qt_data_drive.items_hidden_by_mount", return_value=3
        ):
            self.select(dialog, "/dev/sda2")
            dialog.folder_edit.setText("/mnt/old")
            self.assertIn("already holds 3 item(s) on the system disk", dialog.folder_note.text())
            self.assertTrue(dialog.mount_button.isEnabled())  # a warning, not a refusal

    def mounted_setup(self, *, kept: bool) -> None:
        next(d for d in self.devices if d.path == "/dev/sda2").mountpoint = DATA_FOLDER
        self.mounted = {DATA_FOLDER}
        if kept:
            self.fstab = FSTAB_SYSTEM + f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults,nofail 0 0\n"

    def test_keep_a_mounted_drive_after_a_restart(self) -> None:
        self.mounted_setup(kept=False)
        with self.dialog() as dialog:
            self.assertFalse(dialog.mounted_page.isHidden())
            self.assertTrue(dialog.own_disk_button.isHidden())
            self.assertIn("not mounted again after a restart", dialog.startup_label.text())
            dialog.startup_button.click()
            self.assertEqual(self.launched[0][5:7], ["storage", "setup"])
            self.assertEqual(self.launched[0][-2:], ["--device", "/dev/sda2"])
            self.fstab = FSTAB_SYSTEM + f"UUID=1234-ABCD {DATA_FOLDER} exfat defaults,nofail 0 0\n"
            dialog._action_finished(0, "ok\n")
            self.assertEqual(self.choices_saved, [(None, True)])
            self.assertEqual(dialog.startup_button.text(), "Stop Mounting at Startup")
            self.assertIn("again after every restart", dialog.result_label.text())

    def test_stop_mounting_at_startup_asks_first(self) -> None:
        self.mounted_setup(kept=True)
        with self.dialog(confirm=False) as dialog:
            dialog.startup_button.click()
            self.assertEqual(self.launched, [])
        with self.dialog() as dialog:
            dialog.startup_button.click()
            self.assertEqual(self.launched[0][5:7], ["storage", "forget"])
            self.assertNotIn("--device", self.launched[0])
            self.fstab = FSTAB_SYSTEM
            dialog._action_finished(0, "ok\n")
            self.assertEqual(dialog.startup_button.text(), "Mount at Every Startup")
            # Still a box that records to a drive: a run without one is refused after the restart.
            self.assertEqual(self.choices_saved, [(None, True)])

    def test_startup_set_for_another_drive_is_flagged(self) -> None:
        self.mounted_setup(kept=False)
        self.fstab = FSTAB_SYSTEM + f"UUID=9999-0000 {DATA_FOLDER} ext4 defaults,nofail 0 2\n"
        with self.dialog() as dialog:
            self.assertIn("different drive", dialog.startup_label.text())
            self.assertEqual(dialog.startup_button.text(), "Use This Drive at Startup")

    def test_safely_remove_the_drive(self) -> None:
        self.mounted_setup(kept=True)
        with self.dialog(confirm=False) as dialog:
            dialog.remove_button.click()
            self.assertEqual(self.launched, [])
        with self.dialog() as dialog:
            dialog.remove_button.click()
            self.assertEqual(self.launched[0][5:7], ["storage", "unmount"])
            next(d for d in self.devices if d.path == "/dev/sda2").mountpoint = ""
            self.mounted = set()
            dialog._action_finished(0, "Unmounted\n")
            self.assertIn("can be unplugged", dialog.result_label.text())
            self.assertFalse(dialog.no_drive_page.isHidden())
            # The drive startup expects is offered again, ready to be mounted back.
            self.assertEqual(dialog.drive_group.checkedButton().property("devicePath"), "/dev/sda2")
            self.assertIn("drive startup is set for", dialog.drive_note.text())

    def test_recording_to_the_system_disk_on_purpose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = str(Path(directory) / "BumbleBoxData")
            with mock.patch("bumblebox_v2.qt_data_drive.own_disk_folder", return_value=folder):
                with self.dialog(confirm=False) as dialog:
                    dialog.own_disk_button.click()
                    self.assertEqual(self.choices_saved, [])
                with self.dialog() as dialog:
                    dialog.own_disk_button.click()
                    self.assertEqual(self.choices_saved, [(folder, False)])
                    self.assertTrue(Path(folder).is_dir())
                    self.assertIn("saved on the system disk", dialog.result_label.text())
                    self.assertEqual(dialog.folder_edit.text(), folder)

    def test_the_window_cannot_be_closed_mid_mount(self) -> None:
        with self.dialog() as dialog:
            dialog.show()
            self.select(dialog, "/dev/sda2")
            dialog.mount_button.click()
            dialog.reject()
            self.assertTrue(dialog.isVisible())
            dialog._action_finished(1, "Storage setup failed: no\n")
            dialog.reject()
            self.assertFalse(dialog.isVisible())


@unittest.skipIf(QApplication is None, "PyQt5 is not available")
class MainWindowDriveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def status(storage):
        return box_status.BoxStatus(
            storage=storage,
            schedule=box_status.ScheduleStatus(box_status.SCHEDULE_OFF, box_status.LEVEL_OFF, "Off"),
            checked_at=None,
        )

    @contextmanager
    def window(self):
        ok = box_status.StatusLine(box_status.LEVEL_OK, "Drive mounted · 212 GB free", DATA_FOLDER, free_gb=212.0)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "bumblebox_v2.qt_gui.collect_box_status", return_value=self.status(ok)
        ):
            root = Path(directory)
            config_path = root / "config.yaml"
            config = load_defaults()
            config["setup"]["completed"] = True
            config["system"]["data_root"] = str(root / "data")
            save_config(config_path, config)
            window = BumbleBoxQtGUI(config_path)
            window._status_timer.stop()
            try:
                yield window, root
            finally:
                window.close()

    def test_the_storage_card_offers_to_mount_a_missing_drive(self) -> None:
        with self.window() as (window, _root):
            window._apply_box_status(
                self.status(box_status.StatusLine(box_status.LEVEL_FAIL, box_status.DRIVE_NOT_MOUNTED, "refused"))
            )
            self.assertEqual(window.data_drive_button.text(), "Mount Data Drive...")
            self.assertEqual(window.data_drive_button.objectName(), "Accent")
            window._apply_box_status(
                self.status(box_status.StatusLine(box_status.LEVEL_OK, "Drive mounted · 212 GB free", DATA_FOLDER))
            )
            self.assertEqual(window.data_drive_button.text(), "Data Drive...")
            self.assertEqual(window.data_drive_button.objectName(), "Quiet")

    def test_the_button_opens_the_window_for_this_config(self) -> None:
        with self.window() as (window, _root), mock.patch.object(qt_gui, "DataDriveDialog") as dialog_class:
            window.data_drive_button.click()
            dialog_class.assert_called_once()
            kwargs = dialog_class.call_args.kwargs
            self.assertEqual(kwargs["config_path"], window.config_path)
            self.assertEqual(kwargs["camera_label"], "")
            dialog_class.return_value.exec_.assert_called_once()

    def test_the_choice_is_saved_and_shown(self) -> None:
        with self.window() as (window, root):
            returned = window._save_drive_choice("/mnt/bees", True)
            saved = load_config(window.config_path)
            self.assertEqual(saved["system"]["data_root"], "/mnt/bees")
            self.assertIs(saved["system"]["require_mounted_storage"], True)
            self.assertEqual(window.config["system"]["data_root"], "/mnt/bees")
            self.assertIs(returned, window.config)
            window._save_drive_choice(None, False)
            self.assertIs(load_config(window.config_path)["system"]["require_mounted_storage"], False)
            self.assertEqual(load_config(window.config_path)["system"]["data_root"], "/mnt/bees")

    def test_a_second_camera_sharing_the_folder_follows(self) -> None:
        from bumblebox_v2.multi_camera import create_second_camera, second_camera_config_path

        with self.window() as (window, root):
            create_second_camera(window.config_path, colony_id="07", data_root=str(root / "data"), cameras=[])
            window.refresh_from_config()
            self.assertEqual(len(window.cameras), 2)
            window._save_drive_choice("/mnt/bees", True)
            second = load_config(second_camera_config_path(window.config_path))
            self.assertEqual(second["system"]["data_root"], "/mnt/bees")
            self.assertIs(second["system"]["require_mounted_storage"], True)

    def test_a_second_camera_with_its_own_folder_is_left_alone(self) -> None:
        from bumblebox_v2.multi_camera import create_second_camera, second_camera_config_path

        with self.window() as (window, root):
            create_second_camera(window.config_path, colony_id="07", data_root=str(root / "other"), cameras=[])
            window.refresh_from_config()
            window._save_drive_choice("/mnt/bees", True)
            second = load_config(second_camera_config_path(window.config_path))
            self.assertEqual(second["system"]["data_root"], str(root / "other"))
            self.assertEqual(second["system"]["require_mounted_storage"], "auto")

    def test_the_drive_is_not_changed_while_something_is_recording(self) -> None:
        with self.window() as (window, _root):
            with mock.patch.object(qt_gui, "capture_in_progress", return_value=False):
                self.assertIsNone(window._drive_busy_reason())
            with mock.patch.object(qt_gui, "capture_in_progress", return_value=True):
                self.assertIn("recording is in progress", window._drive_busy_reason())
            running = self.status(box_status.StatusLine(box_status.LEVEL_OK, "Drive mounted", DATA_FOLDER))
            running.schedule.timers = [SimpleNamespace(running=True, last_started=None)]
            window._camera_statuses = {1: running}
            with mock.patch.object(qt_gui, "capture_in_progress", return_value=False):
                self.assertIn("scheduled recording is running", window._drive_busy_reason())


if __name__ == "__main__":
    unittest.main()
