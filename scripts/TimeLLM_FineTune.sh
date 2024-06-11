#!/bin/bash

# Set the number of products to process sequencially in each batch
BATCH_SIZE=25
TOTAL_PRODUCTS=1000

num_process=1
master_port=00098

echo "Fine Tune 1000 Product true promotions 26 sequence length 4 weeks predictions Pretrained on sales data from store lr=0.001 batch_size = 24" | tee -a $LOGFILE
# Log file where outputs will be saved
LOGFILE="Logfiles/Finetuning_output-26-1000.log"
for (( start=0; start<TOTAL_PRODUCTS; start+=BATCH_SIZE ))
do
    echo "Processing products from $((start + 1)) to $((start + BATCH_SIZE))" | tee -a $LOGFILE
    
    # Run the training script with the appropriate start index and number of products
    accelerate launch --num_processes $num_process --main_process_port $master_port run_few_shot.py --start_product $start --num_products $((start + BATCH_SIZE))| tee -a $LOGFILE
done

    echo "FineTune up to $((start + BATCH_SIZE)) products is done" | tee -a $LOGFILE
