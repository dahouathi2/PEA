model_name=TimeLLM
train_epochs=2
learning_rate=0.001
llama_layers=32

master_port=00098
num_process=1
batch_size=1
d_model=32
d_ff=128

comment='EAN_Channel'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_ean_sequential_vertex.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/true_promo \
  --data_path train.csv \
  --model_id promo_ean_channel \
  --model $model_name \
  --data promo_ean_channel \
  --features MS \
  --target sold_units \
  --patch_len 1 \
  --pred_len 17 \
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
