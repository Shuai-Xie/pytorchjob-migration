import os
import time
import warnings
from threading import Thread

import torch

__all__ = ['MigratableVariable']

# ckpt
ckpt_path = os.environ.get('MIGRATION_CKPT_PATH', './ckpts/migration.pth')
signal_path = os.environ.get('MIGRATION_SIGNAL_PATH', './ckpts/signal')
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_path), exist_ok=True)

# DDP configs
rank = 0
map_location = 'cpu'


def curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class Migratable:
    _instance = None
    migrations = []
    count = 0
    resume = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        cls.count += 1  # count the variable number
        return cls._instance

    def __init__(self, var):
        idx = self.count - 1
        if self.resume and self.count <= len(self.migrations):  # load old var
            if hasattr(var, 'state_dict'):
                var.load_state_dict(self.migrations[idx])
            elif isinstance(var, dict):
                var.update(self.migrations[idx])
                print('load dict:', var)
            else:
                warnings.warn(
                    'only support dict now, this value cannot be recoverd')
            self.migrations[idx] = var
        else:
            self.migrations.append(var)  # add new val


def MigratableVariable(x):
    if Migratable.count == 0:  # 1st invoke
        if not os.path.exists(signal_path):
            write_signal('init')  # write the initial value if not exists

        # load ckpt if exits
        # note: can't be in thread, we must ensure ckpt is loaded before Migratable(x)
        if read_signal() == 'resume':
            print(curtime(), 'loading ckpt...')
            load_ckpt()
            write_signal('init')
            print(curtime(), 'loaded ckpt!')

        if rank == 0:
            listen_signal()

    Migratable(x)
    return x


"""signals"""


def read_signal():
    with open(signal_path, 'r') as f:
        signal = f.readline()
        return signal


def write_signal(signal: str):
    with open(signal_path, 'w') as f:
        f.write(signal)


def prestop_signal(interval=1):
    write_signal('save')
    while True:
        if read_signal() == 'resume':
            print(curtime(), 'saved ckpt!')
            break
        print(curtime(), 'saving ckpt...')
        time.sleep(interval)


def listen_signal(interval=1):
    def listen():
        while True:
            if read_signal() == 'save':
                print(curtime(), 'saving ckpt...')
                save_ckpt()
                write_signal('resume')
                print(curtime(), 'saved ckpt!')  # exit thread
                # break  # make this keep listening and exit with main thread
            time.sleep(interval)

    th = Thread(target=listen, daemon=True)
    th.start()


"""ckpts"""


def save_ckpt():
    ckpts = []
    for v in Migratable.migrations:
        if hasattr(v, 'state_dict'):
            ckpts.append(v.state_dict())
        elif isinstance(v, dict):
            ckpts.append(v)
        else:
            warnings.warn(
                'only support dict now, this value cannot be recoverd')
    torch.save(ckpts, ckpt_path)


def load_ckpt():
    ckpt = torch.load(ckpt_path, map_location=map_location)
    Migratable.migrations = ckpt
    Migratable.resume = True
