# pytorchjob-migration

This repo provides a solution to migrate pytorchjob tasks to newly scheduled nodes and GPUs.



Training a task on k8s may face unexpected fault like the endpoint nodes breaking down suddenly. **Fault-toleration like seamless job migration is critical for training tasks safely on k8s, which can also helps use resources more efficiently by re-scheduling jobs when necessary.**

However, container dynamic migration is not mature now, as discussed in this pytorch-operator [issue #356](https://github.com/kubeflow/pytorch-operator/issues/356). 

To go around this problem, we try to trigger a checkpoint event when the pytrochjob master pod is in `Terminating`  status. This is realized with the help of `preStop` container hook.



## Usage

To use this migration feature, users need to register migration variables in the beginning. Currenly, we only support pass-by-reference types like `dict`.

For example, `model`, `optimizer` and `metrics` in the following code are migratable variables, which can be recovered when job migration occurs.

- Single Process

```python
# metircs to be recorded
metircs = {'epoch': -1, 'best_epoch': -1, 'best_acc': 0.}

# migration
from migration import MigratableVariable
model = MigratableVariable(model)
optimizer = MigratableVariable(optimizer)
metircs = MigratableVariable(metircs)
```

- Multiple Process (DDP)

```python
# metircs to be recorded
metircs = {'epoch': -1, 'best_epoch': -1, 'best_acc': 0.}

# migration
import migration
migration.rank = args.rank  # record rank

from migration import MigratableVariable
model = MigratableVariable(model)
optimizer = MigratableVariable(optimizer)
metircs = MigratableVariable(metircs)
```


Note:

- This solution implements `MigratableVariable` with combination function and singleton class, which can be used in multiple python modules.
- Also, users don't need care when to save and load the checkpoint. Because
  - `save` will be triggered automatically by `preStop` container hook,
  - `load` will be triggered automatically when relaunch the pytorchjob.


## Example

start a common pytorchjob

```sh
$ kubectl apply -f yamls/mnist_ddp_launch.yaml
```

Suddenly, we need migrate (delete + apply) the pytorchjob from 48 to 49.

- save

```sh
# delete pytorchjob
$ kubectl delete -f yamls/mnist_ddp_launch.yaml

# pytorchjob pods are in Terminating statu and an event is triggered to save checkpoint in pvc.
NAME                    READY   STATUS        RESTARTS   AGE     IP              NODE             
mnist-launch-master-0   1/1     Terminating   0          3m18s   10.100.59.175   gpu-10-252-192-48
mnist-launch-worker-0   1/1     Terminating   0          3m18s   10.100.59.131   gpu-10-252-192-48
mnist-launch-worker-1   1/1     Terminating   0          3m18s   10.100.59.136   gpu-10-252-192-48
mnist-launch-worker-2   1/1     Terminating   0          3m18s   10.100.59.132   gpu-10-252-192-48

# training stops
Test Epoch: 0, acc=80.5900
test acc: 80.59, best acc: 80.59
save ckpt at epoch 0
Train Epoch: 1 [0/59]   loss=0.5095
Train Epoch: 1 [10/59]  loss=0.4916
Train Epoch: 1 [20/59]  loss=0.4436
Train Epoch: 1 [30/59]  loss=0.3948
Train Epoch: 1 [40/59]  loss=0.5298
2021-09-26 14:50:18 saving ckpt...      # checkpoint event is triggered
2021-09-26 14:50:19 saved ckpt!
Train Epoch: 1 [50/59]  loss=0.3772
Train Epoch: 1 [58/59]  loss=0.3754

# migration state we save in pvc
-rw-r--r-- 1 root root 89505703 Sep 26 22:50 migration.pth
-rw-r--r-- 1 root root       67 Sep 26 22:50 preStop.log
-rw-r--r-- 1 root root        6 Sep 26 22:50 signal

$ cat preStop.log
2021-09-26 14:50:18 saving ckpt...
2021-09-26 14:50:19 saved ckpt!

$ cat signal
resume
```

- resume

```sh
# apply pytorchjob again
$ kubectl apply -f yamls/mnist_ddp_launch.yaml

# pytorchjob pods are migrated to 49.
NAME                    READY   STATUS    RESTARTS   AGE     IP              NODE             
mnist-launch-master-0   1/1     Running   0          3m19s   10.100.59.5     gpu-10-252-192-49
mnist-launch-worker-0   1/1     Running   1          3m19s   10.100.59.12    gpu-10-252-192-49
mnist-launch-worker-1   1/1     Running   2          3m19s   10.100.59.63    gpu-10-252-192-49
mnist-launch-worker-2   1/1     Running   3          3m19s   10.100.59.24    gpu-10-252-192-49

# and training resumes.
2021-09-26 14:52:04 loading ckpt...
2021-09-26 14:52:04 loaded ckpt!
load dict: {'epoch': 0, 'best_epoch': 0, 'best_acc': 80.59}

training
Train Epoch: 1 [0/59]   loss=0.4146
Train Epoch: 1 [10/59]  loss=0.4130
Train Epoch: 1 [20/59]  loss=0.3693
Train Epoch: 1 [30/59]  loss=0.3137
Train Epoch: 1 [40/59]  loss=0.4815
Train Epoch: 1 [50/59]  loss=0.3160
Train Epoch: 1 [58/59]  loss=0.2624
Test Epoch: 1 [0/40]    acc=86.3281
Test Epoch: 1 [10/40]   acc=85.7244
Test Epoch: 1 [20/40]   acc=85.4539
Test Epoch: 1 [30/40]   acc=85.4461
Test Epoch: 1 [39/40]   acc=85.5300
Test Epoch: 1, acc=85.5300
test acc: 85.53, best acc: 85.53
save ckpt at epoch 1
```
