# For compression with a constant replacing rate
export GLUE_DIR=/home/lilei/transformers/glue_data
export TASK_NAME=SST-2
export TASK_DATA="sst-2"
export GPU="0" # ,1,2,3" # ,1,2,3" #,1,2,3"
 #   $PRE_PATH  \
EPOCH=10
BSZ=32
for SEED in 1234 #2345 3456
do
for pr in 0.0  0.001 0.005 0.01 0.05 # 0.1  0.2 0.5  2.0 5.0 10.0  # 10.0 50.0 100.0 1000.0   
do
#for idx in $(seq 0 63) # 62 60 56 48 32 63  0 1 3 7 15 31 
#do
# train theseus bert
#for pidx in 0 63 
#do
PRE_PATH=/home/lilei/transformers/successor_result/${TASK_DATA}_output_$SEED/
OUTPUT_PATH=results/${TASK_NAME}_theseus_linear_${SEED}_bsz64 
CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path  ${OUTPUT_PATH} \
  --task_name $TASK_NAME --seed $SEED --fp16 --switch_mode --fix_scc_layer --early_exit  \
  --do_eval --do_train \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ \
  --learning_rate 2e-5 \
  --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs $EPOCH --path_penalty_ratio $pr \
  --output_dir  ${OUTPUT_PATH}_nas_penalty_ratio${pr}_epoch${EPOCH}_fix_scc_early_exit/ \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006
#done 
done
done



