'''stop the automated recording schedule WITHOUT replacing it with a new schedule, by removing the commands from the crontab'''

import setup
from crontab import CronTab
import os
import pwd
import glob


def detect_storage_device():
    configured = getattr(setup, "storage_device", "/dev/sda1")
    if configured and os.path.exists(configured):
        return configured
    for by_label in glob.glob("/dev/disk/by-label/*"):
        if os.path.basename(by_label).lower() == "bumblebox":
            return by_label
    return configured or "/dev/sda1"


def main():
    username = pwd.getpwuid(os.getuid())[0]
    mount_point = getattr(setup, "storage_mount_point", setup.data_folder_path)
    storage_device = detect_storage_device()

    cron = CronTab(user=username)
    cron.remove_all()

    job1 = cron.new(command=f"sudo mount {storage_device} {mount_point} -o umask=000")
    job1.every_reboot()

    job2 = cron.new(command=f"sudo chmod -R ugo+rwx {mount_point}")
    job2.every_reboot()

    cron.write()


if __name__ == "__main__":
    main()
