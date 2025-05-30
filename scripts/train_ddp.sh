CONFIG=$1

GPUS_PER_NODE=${GPUS_PER_NODE:-8}   # number of gpus per node
NNODES=${NNODES:-1}            # Number of total nodes (default: 1).
NODE_RANK=${NODE_RANK:-0}      # Rank of the current node (default: 0).
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  # IP address of the master node.
MASTER_PORT=${MASTER_PORT:-29500}            # Port used for communication (default: 29500).


torchrun --nproc_per_node=$GPUS_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         scripts/train.py \
         $CONFIG \
         --launcher pytorch ${@:2}