# SP (Single Process)
CUDA_VISIBLE_DEVICES=0 python mnist_dp.py --epochs=10 --batch-size=256

# DP
python mnist_dp.py --epochs=10 --batch-size=256

# DDP launch.py
# 1 nodes 1m
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=10 --batch-size=256
# 2 nodes 2m
# host 执行，要求每台机器 nproc_per_node 相等，不然计算的 world_size 不同
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
# 47/48/49 connects 38/42/43 NCCL error ??

# 4 nodes 2m
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=0 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=1 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=2 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=3 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256

# 8 nodes 2m
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=0 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=1 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=2 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=3 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=4 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=5 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=6 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=8 --node_rank=7 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256

# 2 nodes 1m
# note: must set CUDA_VISIBLE_DEVICES explicitly to assign gpus like pods to avoid NCCL error
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256


# DDP mp.spawn tcp
# 1 nodes 1m
python mnist_ddp_mp.py --nproc_per_node=4 --nnodes=1 --node_rank=0 --dist-url="tcp://10.252.192.48:22222" --epochs=10 --batch-size=256
# 2 nodes 2m
python mnist_ddp_mp.py --nproc_per_node=4 --nnodes=2 --node_rank=0 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
python mnist_ddp_mp.py --nproc_per_node=4 --nnodes=2 --node_rank=1 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
# 4 nodes 2m
CUDA_VISIBLE_DEVICES=0,1 python mnist_ddp_mp.py --nproc_per_node=2 --nnodes=4 --node_rank=0 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2,3 python mnist_ddp_mp.py --nproc_per_node=2 --nnodes=4 --node_rank=1 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=0,1 python mnist_ddp_mp.py --nproc_per_node=2 --nnodes=4 --node_rank=2 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2,3 python mnist_ddp_mp.py --nproc_per_node=2 --nnodes=4 --node_rank=3 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
# 8 nodes 2m
CUDA_VISIBLE_DEVICES=0 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=0 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=1 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=1 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=2 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=3 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=3 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=0 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=4 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=1 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=5 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=6 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=3 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=7 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256


# DDP mp.spawn file
# 1 nodes
python mnist_ddp_mp.py --nproc_per_node=4 --nnodes=1 --node_rank=0 --dist-url="file:///export/nfs/xs/codes/pytorch_operator_example/sharedfile" --epochs=1 --batch-size=256
# 2 nodes
python mnist_ddp_mp.py --nproc_per_node=4 --nnodes=2 --node_rank=0 --dist-url="file:///export/nfs/xs/codes/pytorch_operator_example/sharedfile"  --epochs=1 --batch-size=256
python mnist_ddp_mp.py --nproc_per_node=4 --nnodes=2 --node_rank=1 --dist-url="file:///export/nfs/xs/codes/pytorch_operator_example/sharedfile"  --epochs=1 --batch-size=256

kill -9 $(ps -ef |grep xs |grep python |awk '{print $2}')