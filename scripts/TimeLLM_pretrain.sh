model_name=TimeLLM
train_epochs=2
learning_rate=0.001
llama_layers=32

master_port=00098
num_process=1
batch_size=24
d_model=32
d_ff=128

comment='TimeLLM-Pretrain-26-8640'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset \
  --data_path ETTh1.csv \
  --model_id Pretrain \
  --model $model_name \
  --data ETTh1 \
  --features MS \
  --target OT \
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
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model GPT2 \
  --llm_dim 768 
