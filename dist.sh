export NCCL_SOCKET_IFNAME=bond0:10.11.1.2
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    dist_demo.py \
