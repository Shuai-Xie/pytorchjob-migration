import time

from migration import Migrator
from utils import curtime

mig = Migrator()
mig.write_signal('save')

while True:
    if mig.resume:
        print(curtime(), 'finish ckpts!')
        break
    print(curtime(), 'saving ckpts...')
    time.sleep(1)
