"""Set up the tag tracking/recording schedule based on setup.py."""

import os
import pwd
import subprocess
import glob

from crontab import CronTab

import setup


def find_file(file_name, directory_path):
    for dirpath, dirnames, fnames in os.walk(directory_path):
        for filename in fnames:
            if filename == file_name:
                return dirpath + "/" + file_name
    return None


def _minutes_for_frequency(frequency):
    if frequency <= 0:
        raise ValueError("Frequency must be a positive integer")
    if frequency >= 60:
        return [0]
    return list(range(0, 60, frequency))


def detect_storage_device():
    configured = getattr(setup, "storage_device", "/dev/sda1")
    if configured and os.path.exists(configured):
        return configured

    for by_label in glob.glob("/dev/disk/by-label/*"):
        if os.path.basename(by_label).lower() == "bumblebox":
            return by_label

    try:
        lsblk_output = subprocess.check_output(
            ["lsblk", "-prno", "NAME,RM,TYPE"],
            text=True,
        )
        for line in lsblk_output.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            name, is_removable, device_type = parts
            if is_removable == "1" and device_type == "part":
                return name
    except Exception:
        pass

    return configured or "/dev/sda1"


def main():
    username = pwd.getpwuid(os.getuid())[0]
    home_dir = f"/home/{username}"
    mount_point = getattr(setup, "storage_mount_point", setup.data_folder_path)
    storage_device = detect_storage_device()

    record_video_path = find_file("record_video.py", home_dir) or os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "record_video.py",
    )

    cron = CronTab(user=username)
    cron.remove_all()

    job0 = cron.new(command=f"sudo mkdir -p {mount_point}")
    job0.every_reboot()

    job1 = cron.new(command=f"sudo mount {storage_device} {mount_point} -o umask=000")
    job1.every_reboot()

    job2 = cron.new(command=f"sudo chmod -R ugo+rwx {mount_point}")
    job2.every_reboot()

    job3 = cron.new(
        command=(
            f"python3 {record_video_path} --data_folder_path {setup.data_folder_path} "
            f"-t {setup.recording_time} -q {setup.quality} -fps {setup.frames_per_second} "
            f"--shutter {setup.shutter_speed} -w {setup.width} -ht {setup.height} "
            f"-d {setup.tag_dictionary} --box_type {setup.box_type} -cd {setup.codec} "
            f"-tf {setup.tuning_file} -nr {setup.noise_reduction_mode} "
            f"-z {setup.recording_digital_zoom} > /home/{username}/Desktop/output.txt 2>&1"
        )
    )
    job3.minute.every(setup.recording_frequency)

    if setup.tag_tracking is True:
        tag_tracking_path = find_file("ram_capture_tag_tracking.py", home_dir) or os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "ram_capture_tag_tracking.py",
        )

        recording_minutes = set(_minutes_for_frequency(setup.recording_frequency))
        tag_tracking_minutes = _minutes_for_frequency(setup.tag_tracking_frequency)
        tag_tracking_without_recording_minutes = [
            minute for minute in tag_tracking_minutes if minute not in recording_minutes
        ]

        if tag_tracking_without_recording_minutes:
            job4 = cron.new(
                command=(
                    f"python3 {tag_tracking_path} --data_folder_path {setup.data_folder_path} "
                    f"-t {setup.recording_time} -fps {setup.frames_per_second} "
                    f"-afps {setup.actual_frames_per_second} --shutter {setup.shutter_speed} "
                    f"-w {setup.width} -ht {setup.height} -d {setup.tag_dictionary} "
                    f"-tf {setup.tuning_file} --box_type {setup.box_type} "
                    f"-nr {setup.noise_reduction_mode} -z {setup.recording_digital_zoom} "
                    f"> /home/{username}/Desktop/output.txt 2>&1"
                )
            )
            job4.minute.on(tag_tracking_without_recording_minutes[0])
            for minute in tag_tracking_without_recording_minutes[1:]:
                job4.minute.also.on(minute)
        else:
            print("No non-overlapping tag-tracking minutes found; tracking cron job not added.")

    if setup.create_composite_nest_images is True:
        generate_nest_images_path = find_file("generate_nest_images.py", home_dir) or os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "generate_nest_images.py",
        )
        job5 = cron.new(
            command=(
                f"python3 {generate_nest_images_path} --data_folder_path {setup.data_folder_path} "
                f"--number_of_images {setup.number_of_images}"
            )
        )
        job5.setall("0 23 * * *")

    cron.write()

    try:
        subprocess.call(["crontab", "-l"])
    except Exception as e:
        print(e)
        print(e.args)
        print("Couldnt print the crontab commands")


if __name__ == "__main__":
    main()
