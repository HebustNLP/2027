CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"

python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=cluener \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/cluener \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=1e-5 \
  --num_train_epochs=15.0 \
  --logging_steps=1344 \
  --save_steps=1344 \
  --output_dir=/root/autodl-tmp/third/outputs \
  --overwrite_output_dir \
  --seed=42





python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=people_daily \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/people_daily \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=15.0 \
  --logging_steps=652 \
  --save_steps=652 \
  --output_dir=/root/autodl-tmp/third/outputs_people_daily \
  --overwrite_output_dir \
  --seed=42


# one-bc 重新跑下
python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=one-bc \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/one/bc \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=1e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=983 \
  --save_steps=983 \
  --output_dir=/root/autodl-tmp/third/outputs_one_bc \
  --overwrite_output_dir \
  --seed=42


# one-bn
python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=one-bn \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/one/bn \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=1e-5 \
  --num_train_epochs=8.0 \
  --logging_steps=1019 \
  --save_steps=1019 \
  --output_dir=/root/autodl-tmp/third/outputs_one_bn \
  --overwrite_output_dir \
  --seed=42  



# one-mz
python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=one-mz \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/one/mz \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=1e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=499 \
  --save_steps=499 \
  --output_dir=/root/autodl-tmp/third/outputs_one_mz \
  --overwrite_output_dir \
  --seed=42  

# one-nw
python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=one-nw \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/one/nw \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=1e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=447 \
  --save_steps=447 \
  --output_dir=/root/autodl-tmp/third/outputs_one_nw \
  --overwrite_output_dir \
  --seed=42  

# one-tc
python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=one-tc \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/one/tc \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=939 \
  --save_steps=939 \
  --output_dir=/root/autodl-tmp/third/outputs_one_tc \
  --overwrite_output_dir \
  --seed=42


# one-wb
python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=one-wb \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/one/wb \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=6.0 \
  --logging_steps=810 \
  --save_steps=810 \
  --output_dir=/root/autodl-tmp/third/outputs_one_wb \
  --overwrite_output_dir \
  --seed=42

# weibo
python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=/root/autodl-tmp/third/prev_trained_model/bert \
  --task_name=weibo \
  --do_train \
  --do_eval \
  --warmup_steps=338 \
  --weight_decay=0.01 \
  --do_lower_case \
  --loss_type=lsr \
  --data_dir=/root/autodl-tmp/third/data/weibo \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=15.0 \
  --logging_steps=507 \
  --save_steps=507 \
  --output_dir=/root/autodl-tmp/third/outputs_weibo \
  --overwrite_output_dir \
  --seed=42