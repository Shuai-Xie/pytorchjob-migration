import time

from migration import curtime, migrator

migrator.write_signal('save')

while True:
    if migrator.resume:
        print(curtime(), 'finish ckpts!')
        break
    print(curtime(), 'saving ckpts...')
    time.sleep(1)
