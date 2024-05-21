model_name=TimeLLM
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=8
batch_size=100
d_model=32
d_ff=128

comment='TimeLLM-ETTh1_ETTh2'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset \
  --data_path_pretrain ETTh1.csv \
  --data_path ETTh2.csv \
  --model_id ETTh1_ETTh2_512_96 \
  --model $model_name \
  --data_pretrain ETTh1 \
  --data ETTh2 \
  --features M \
  --seq_len 26 \
  --label_len 4 \
  --pred_len 4 \
  --factor 3 \
 --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 5 \
  --model_comment $comment
  --llm_model GPT2 \
  --llm_dim 768
