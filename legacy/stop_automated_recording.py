'''stop the automated recording schedule WITHOUT replacing it with a new schedule, by removing the commands from the crontab'''

import setup
from crontab import CronTab
import os
import pwd

'''store the computer username for use below'''
username = pwd.getpwuid(os.getuid())[0]

'''make sure the computer recognizes the thumb drive or storage device when the computer turns on, and name it bumblebox'''
cron = CronTab(user=f'{username}')
cron.remove_all()
job1 = cron.new(command='sudo mount /dev/sda1 /mnt/bumblebox -o umask=000')
job1.every_reboot()
cron.write()

'''make sure that we can write to this storage drive and to folders that we create on it'''
job1 = cron.new(command='sudo chmod -R ugo+rwx /mnt/bumblebox')
job1.every_reboot()
cron.write()