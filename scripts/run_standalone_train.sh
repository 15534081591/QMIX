script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
DEVICE_ID=$1
MAP_NAME=$2
python -s ${self_path}/../src/main.py --config=qmix --env-config=sc2 with env_args.map_name=$MAP_NAME device_id=$DEVICE_ID checkpoint_path=''> trainlog.txt 2>&1 &