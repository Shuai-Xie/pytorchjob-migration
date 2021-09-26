# pytorchjob-migration

This repo provides a solution to migrate pytorchjob tasks to newly scheduled nodes and GPUs.



Training a task on k8s may face unexpected fault like the endpoint nodes breaking down suddenly. **Fault-toleration like seamless job migration is critical for training tasks safely on k8s, which can also helps use resources more efficiently by re-scheduling jobs when necessary.**

However, container dynamic migration is not mature now, as discussed in this pytorch-operator [issue #356](https://github.com/kubeflow/pytorch-operator/issues/356). 

To go around this problem, we try to trigger a checkpoint event when a pytrochjob pod in `Terminating`  status, which is realized with `preStop` container hook.



## Usage

To use this migration feature, users need to register migration variables in the beginning. Currenly, we only support pass-by-reference types like `dict` or `list`.

For example, `model`, `optimizer` and `metrics` in the following code are migration variables, which can be recovered when job migration occurs.

```python
# metircs to be recorded
metircs = {'epoch': -1, 'best_epoch': -1, 'best_acc': 0.}

# A migrator helps training models on k8s more secure.
from migration import migrator
migrator.register('model', model)
migrator.register('optimizer', optimizer)
migrator.register('metircs', metircs)
migrator.listening()
if migrator.resume:  # note: migrate_ckpt has higher priority than args.ckpt
    migrator.load_ckpt()  # load ckpt at all ranks
    print('load migration ckpt from epoch {}, metrics: {}'.format(
        metircs['epoch'], metircs))
```

Note:

- This solution imports a singleton migrator to register values, which can be reused in other python modules without instantiation.
- However, the limitation is that `migrator.load_ckpt()` have to be invoked after all the migration variables have been registered.


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
Train Epoch: 4 [0/59]   loss=0.2813
Train Epoch: 4 [10/59]  loss=0.2778
Train Epoch: 4 [20/59]  loss=0.2248
Train Epoch: 4 [30/59]  loss=0.3184
Train Epoch: 4 [40/59]  loss=0.2645
Train Epoch: 4 [50/59]  loss=0.3380
Train Epoch: 4 [58/59]  loss=0.2972
Test Epoch: 4 [0/40]    acc=88.2812
2021-09-24 11:07:58 save migrate ckpt		# checkpoint event is triggered
2021-09-24 11:07:59 saved ckpt!
Test Epoch: 4 [10/40]   acc=86.0085
Test Epoch: 4 [20/40]   acc=86.1049

# migration state we save in pvc
-rw-r--r-- 1 root root 89505767 Sep 24 19:07 migration.pth
-rw-r--r-- 1 root root       36 Sep 24 19:04 postStart.log
-rw-r--r-- 1 root root      105 Sep 24 19:07 preStop.log
-rw-r--r-- 1 root root        6 Sep 24 19:07 signal

$ cat postStart.log
start: Fri Sep 24 11:04:45 UTC 2021

$ cat preStop.log
stop: Fri Sep 24 11:07:57 UTC 2021
2021-09-24 11:07:58 saving ckpts...
2021-09-24 11:07:59 finish ckpts!

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
load migration ckpt from epoch 4, metrics: {'epoch': 4, 'best_epoch': 3, 'best_acc': 85.56}

training
Train Epoch: 5 [0/59]   loss=0.3343
Train Epoch: 5 [10/59]  loss=0.3286
Train Epoch: 5 [20/59]  loss=0.2807
Train Epoch: 5 [30/59]  loss=0.3667
Train Epoch: 5 [40/59]  loss=0.3388
Train Epoch: 5 [50/59]  loss=0.3381
Train Epoch: 5 [58/59]  loss=0.3596
Test Epoch: 5 [0/40]    acc=86.3281
Test Epoch: 5 [10/40]   acc=86.0440
Test Epoch: 5 [20/40]   acc=85.3051
Test Epoch: 5 [30/40]   acc=85.3075
Test Epoch: 5 [39/40]   acc=85.4500
Test Epoch: 5, acc=85.4500
test acc: 85.45, best acc: 85.45
save ckpt at epoch 5
```
