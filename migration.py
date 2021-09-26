import os
import time
import warnings
from threading import Thread

import torch


def curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class Migrator:
    """
    Migration is triggered by three singals:
    1. init: start listening.
    2. save: prestop writes save signal, save ckpt and exits.
    3. resume: main reads resume signal, load ckpt and continue training.

    We need save signal to file, cus we need this signal when relaunching training.
    If signal is a variable not read from file, we'll lose the signal set by last training.
    """
    def __init__(self) -> None:
        self.ckpt_path = os.environ.get('MIGRATION_CKPT_PATH',
                                        './ckpts/migration.pth')
        self.signal_path = os.environ.get('MIGRATION_SIGNAL_PATH',
                                          './ckpts/signal')
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.signal_path), exist_ok=True)
        if not os.path.exists(self.signal_path):
            self.write_signal('init')  # only init at 1st time
        self.migrations = {}

    @property
    def signal(self):  # 同一 signal 文件路径，1个可供多进程操作的单例属性
        return self.read_signal()

    @property
    def resume(self):
        return self.signal == 'resume'

    def write_signal(self, signal: str):
        with open(self.signal_path, 'w') as f:
            f.write(signal)

    def read_signal(self):
        with open(self.signal_path, 'r') as f:
            signal = f.readline()
            return signal

    def register(self, key: str, val):
        self.migrations[key] = val

    def _listen(self, interval=1):  # signal 状态机
        while True:
            if self.signal == 'save':  # save -> save_ckpt() -> resume
                print(curtime(), 'save migrate ckpt')
                self.save_ckpt()
                print(curtime(), 'saved ckpt!')
                break
            # 为了确保 load_ckpt 在 main process 使用前执行；将 resume 过程放到 main process
            # elif signal == 'resume':  # resume -> load_ckpt() -> init
            #     print(curtime(), 'resume migrate ckpt')
            #     self.load_ckpt()
            time.sleep(interval)

    def listening(self, interval=1):
        self.th = Thread(target=self._listen, args=(interval, ), daemon=True)
        self.th.start()

    def save_ckpt(self):
        for k, v in self.migrations.items():
            if hasattr(v, 'state_dict'):
                self.migrations[k] = v.state_dict()
        torch.save(self.migrations, self.ckpt_path)
        self.write_signal('resume')

    def load_ckpt(self, map_location='cpu'):
        ckpt = torch.load(self.ckpt_path, map_location=map_location)
        for k, v in self.migrations.items():
            if hasattr(v, 'state_dict'):
                self.migrations[k].load_state_dict(ckpt[k])
            elif isinstance(v, dict):
                self.migrations[k].update(ckpt[k])  # only support dict
            else:
                warnings.warn(
                    'only support dict now, this value cannot be recoverd')
                print(k, v)
        self.write_signal('init')


# singleton varible
migrator = Migrator()
