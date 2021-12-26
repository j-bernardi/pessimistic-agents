if [ "$#" -ne 1 ]; then
    echo "Must run with only 1 arg, config ID"
    exit 0
fi

NUM_DEVICES=4
CFG_ID=$1

DEVICE_NUM=$((CFG_ID % NUM_DEVICES))
ADDRESS=$((8476 + DEVICE_NUM))

export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1

export TPU_VISIBLE_DEVICES=$DEVICE_NUM
export TPU_MESH_CONTROLLER_ADDRESS=localhost:$ADDRESS TPU_MESH_CONTROLLER_PORT=$ADDRESS

echo "RUNNING python3 experiments/core_experiment/finite_agent_0.py -c $CFG_ID"
echo "ON $TPU_VISIBLE_DEVICES, $ADDRESS"
python3 experiments/core_experiment/finite_agent_0.py -c $CFG_ID
