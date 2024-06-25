#!/usr/bin/env bash
set -x
cd ..

# used GPUs
GPUS='0'
GPUS_PER_NODE=1

CPUS_PER_TASK=6
PORT=29555
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="swin_l_p4w7"
OUTPUT_DIR="../results/HTR_${BACKBONE}_eval"
CHECKPOINT="Your path to the model weights.pth"
python inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
  --eval \
  --ngpu=${GPUS_PER_NODE} \
  --output_dir=${OUTPUT_DIR} \
  --resume=${CHECKPOINT} \
  --backbone=${BACKBONE} \
  --amp \




