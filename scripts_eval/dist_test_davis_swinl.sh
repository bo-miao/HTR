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
python inference_davis.py --with_box_refine --binary --freeze_text_encoder \
  --eval \
  --ngpu=${GPUS_PER_NODE} \
  --output_dir=${OUTPUT_DIR} \
  --resume=${CHECKPOINT} \
  --backbone=${BACKBONE} \
  --amp \

# evaluation
ANNO0_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_0"
rm ${ANNO0_DIR}"/global_results-val.csv"
rm ${ANNO0_DIR}"/per-sequence_results-val.csv"
python3 eval_davis.py --results_path=${ANNO0_DIR}
echo "Annotations store at : ${ANNO0_DIR}"

ANNO1_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_1"
rm ${ANNO1_DIR}"/global_results-val.csv"
rm ${ANNO1_DIR}"/per-sequence_results-val.csv"
python3 eval_davis.py --results_path=${ANNO1_DIR}

ANNO2_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_2"
rm ${ANNO2_DIR}"/global_results-val.csv"
rm ${ANNO2_DIR}"/per-sequence_results-val.csv"
python3 eval_davis.py --results_path=${ANNO2_DIR}

ANNO3_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_3"
rm ${ANNO3_DIR}"/global_results-val.csv"
rm ${ANNO3_DIR}"/per-sequence_results-val.csv"
python3 eval_davis.py --results_path=${ANNO3_DIR}

echo "Working path is: ${OUTPUT_DIR}"



