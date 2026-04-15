set -ex

export CUDA_VISIBLE_DEVICES=0

# ======================
# 固定变量（与训练时一致）
# ======================
K=1
CTRL_TOKEN=none
DATA_RATIO=1.0

declare -A TASK_DATA
TASK_DATA[asqp]="R15 R16"
# TASK_DATA[acos]="Lap Rest"
TASK_DATA[acos]="Rest"

cd src

# ======================
# 只跑推理（不训练）
# ======================
for TASK in acos
do
for DATA in ${TASK_DATA[${TASK}]}
do
for SEED in 10
do

OUT_DIR="../outputs/$TASK/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}"

# ⚠️ 要求 OUT_DIR 里已经存在：
#   OUT_DIR/first
#   OUT_DIR/final



echo ">>> Inference with pretrained model at: $OUT_DIR"

python main.py \
    --data_path "../data/" \
    --dataset $DATA \
    --model_name ../outputs/acos/Rest/top1_none_data1.0/final \
    --task $TASK \
    --output_dir $OUT_DIR \
    --eval_batch_size 64 \
    --seed $SEED \
    --lowercase \
    --sort_label \
    --constrained_decode \
    --do_inference \
    --eval_data_split test \
    | tee ${OUT_DIR}/infer.log \
    2> ${OUT_DIR}/infer.err

done
done
done
