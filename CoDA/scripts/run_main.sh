set -ex

export CUDA_VISIBLE_DEVICES=0


# 补充缺失的变量定义
K=1  # 例如选择top 1的视图
CTRL_TOKEN=none  # 控制令牌类型，根据任务需求设置
DATA_RATIO=1.0   # 数据集比例，1.0表示使用全部数据

declare -A TASK_DATA
TASK_DATA[asqp]="R16 R15"
#TASK_DATA[acos]="Lap Rest"
TASK_DATA[acos]="Rest Lap"

cd src

# for SVP_TYPE in heuristic rand rank 
#for TASK in asqp acos memd
for TASK in acos
do
for DATA in ${TASK_DATA[${TASK}]}
do
for SEED in 5 15 35
do
OUT_DIR="../outputs/$TASK/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}"

mkdir -p $OUT_DIR

# ---- auto-detect complexity cache (directory mode) ----
COMPLEX_DIR="../data/${TASK}/${DATA}"
COMPLEX_ARGS=""

shopt -s nullglob
COMPLEX_FILES=("$COMPLEX_DIR"/complexity*.json)
shopt -u nullglob

if [ -d "$COMPLEX_DIR" ] && [ ${#COMPLEX_FILES[@]} -gt 0 ]; then
  echo "[complexity] detected ${#COMPLEX_FILES[@]} files under: $COMPLEX_DIR"
  COMPLEX_ARGS="--enable_complexity_aux --complexity_cache_path ${COMPLEX_DIR}"
else
  echo "[complexity] no complexity*.json under: $COMPLEX_DIR (skip)"
fi

python main.py \
    --data_path "../data/" \
    --dataset $DATA \
    --model_name 模型路径 \
    --output_dir $OUT_DIR \
    --num_train_epochs 30 \
    --save_top_k 1 \
    --task $TASK \
    --first_stage_views exclude_O \
    --seed $SEED \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --struct_with_l \
    --use_struct_token \
    --check_val_every_n_epoch 10  \
    --eval_batch_size 64 \
    --constrained_decode \
    --do_train \
    --loss_lambda 0.02 \
    --cont_temp 0.5 \
    --cont_loss 1.0 \
    --proj_dim 256 \
    --aux_high_merge_only_add \
    $COMPLEX_ARGS \
    --complexity_threshold 1 \
    --aux_templates 2 \
    --aux_vote_threshold 2 \
    --aux_use_stage1_order \
    2>&1 | tee ${OUT_DIR}/train.log
    # --model_name_or_path "PATH TO THE CHECKPOINT" \ # configure the checkpoint path to eval

    # --load_path_cache \
    # --single_view_type $SVP_TYPE \
    # --load_ckpt_name "ckpt path" \
    # > $OUT_DIR/train.log 2>&1&
done
done
done
done
# done
