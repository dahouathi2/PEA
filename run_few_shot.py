import argparse

import torch

from accelerate import Accelerator, DeepSpeedPlugin

from accelerate import DistributedDataParallelKwargs

from torch import nn, optim

from torch.optim import lr_scheduler

from tqdm import tqdm



from models import Autoformer, DLinear, TimeLLM



#from data_provider.data_factory import data_provider

import time

import random

import numpy as np

import os



os.environ['CURL_CA_BUNDLE'] = ''

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"



from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content


fix_seed = 2021

random.seed(fix_seed)

torch.manual_seed(fix_seed)

np.random.seed(fix_seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')

accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from models import TimeLLM
from data_provider.data_factory import data_provider
import time
import torch

class Args:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'Pretrain'
        self.model_comment = 'TimeLLM-Pretrain-13-2500-finetuning'
        self.model = 'TimeLLM'
        self.seed = 2021
        self.data = 'few_shot_seq'
        self.root_path = './dataset'
        self.data_path = 'true_promo_data_77.csv'
        self.features = 'MS'
        self.target = 'sold_units'
        self.loader = 'modal'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'
        self.seq_len = 26
        self.label_len = 4
        self.pred_len = 4
        self.seasonal_patterns = 'Monthly'
        self.enc_in = 9
        self.dec_in = 9
        self.c_out = 9
        self.d_model = 32
        self.n_heads = 8  # Typically set by your model configuration
        self.e_layers = 2  # Typically set by your model configuration
        self.d_layers = 1  # Typically set by your model configuration
        self.d_ff = 128
        self.moving_avg = 25  # Assume default if not specified in the script
        self.factor = 3
        self.dropout = 0.1  # Assume default if not specified
        self.embed = 'timeF'  # Assume default if not specified
        self.activation = 'gelu'  # Assume default if not specified
        self.output_attention = False  # Assume default if not specified
        self.patch_len = 16  # Assume default if not specified
        self.stride = 8  # Assume default if not specified
        self.prompt_domain = 0  # Assume default if not specified
        self.llm_model = 'GPT2'
        self.llm_dim = 768
        self.num_workers = 10  # Default setting
        self.itr = 1
        self.train_epochs = 5
        self.align_epochs = 10  # Assume default if not specified
        self.batch_size = 1
        self.eval_batch_size = 8  # Assume default if not specified
        self.patience = 10  # Assume default if not specified
        self.learning_rate = 0.001
        self.des = 'Exp'
        self.loss = 'MSE'  # Assume default if not specified
        self.lradj = 'type1'  # Assume default if not specified
        self.pct_start = 0.2  # Assume default if not specified
        self.use_amp = False  # Assume default based on your environment capabilities
        self.llm_layers = 32
        self.percent = 100  # Assume default if not specified

# Instantiate the Args
args = Args()
parser = argparse.ArgumentParser(description='Train model on product batches')
parser.add_argument('--start_product', type=int, default=0, help='Start index of products to train')
parser.add_argument('--num_products', type=int, default=25, help='Number of products to train')
args2 = parser.parse_args()
final_index=args2.num_products # purpose of this code is to finetune sequencially on products
start_index=args2.start_product

combined_dataframes = pd.read_csv('./dataset/true_promo_data_77.csv')
if start_index == 0: # we load pretrained model
    print("Success Loading Pretrained Model")
    path = 'checkpoints/long_term_forecast_Pretrain_TimeLLM_pretrain_ftMS_sl26_ll4_pl4_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Exp_0-TimeLLM-Pretrain-26-others-cleaning-produce/checkpoint'
else: # we load latest fine tuning process
    print("Success Loading last Finetuned Model on {} products".format(start_index))
    path = 'checkpoints/Finetune/Finetune_{}.pth'.format(start_index)
# Load the checkpoint
model = TimeLLM.Model(args).float()
model.load_state_dict(torch.load(path), strict=False)

from data_provider.data_loader import Dataset_few_shot_seq
from torch.utils.data import DataLoader

def data_provider(args, flag, data):
    
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test' or flag == 'few_shot': # zero shot not few shot
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Dataset_few_shot_seq(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        seasonal_patterns=args.seasonal_patterns,
        df=data,
        scale = True
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


criterion = nn.MSELoss()
mae_metric = nn.L1Loss()


import gc

# Store the initial learning rate separately
initial_learning_rate = args.learning_rate

for index_product in range(start_index,final_index):
    print("Starting fine-tuning on product {}".format(index_product+1))

    data = combined_dataframes[72 * index_product : 72 * (index_product + 1)]
    train_data, train_loader = data_provider(args, 'train', data)
    vali_data, vali_loader = data_provider(args, 'val', data)
    test_data, test_loader = data_provider(args, 'test', data)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    # Reinitialize optimizer and scheduler
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=initial_learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=initial_learning_rate,
        )

    # Prepare everything with the new optimizer and scheduler
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler
    )

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        model.train()
        iter_count = 0
        train_loss = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            # Forward pass
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -args.pred_len :, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
            else:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -args.pred_len :, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                accelerator.backward(loss)
                model_optim.step()

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                )
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(
                    "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time)
                )
                iter_count = 0
                time_now = time.time()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        with torch.cuda.amp.autocast():
            vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss
            )
        )

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            accelerator.print("Updating learning rate to {}".format(scheduler.get_last_lr()[0]))

    # Free up GPU memory
    del train_data, train_loader, vali_data, vali_loader, test_data, test_loader, dec_inp, batch_x, batch_y, batch_x_mark, batch_y_mark
    torch.cuda.empty_cache()
    gc.collect()


    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        accelerator.print("FineTuning is done for product {}".format(index_product+1))

state = accelerator.unwrap_model(model)
torch.save(state.state_dict(), './checkpoints/Finetune/Finetune_{}.pth'.format(final_index))
previous_model_filename = f'./checkpoints/Finetune/Finetune_{start_index}.pth'

# Delete the previous model if start_index is not a multiple of 100 and not zero
if start_index % 100 != 0 and start_index != 0:
    if os.path.exists(previous_model_filename):
        os.remove(previous_model_filename)
        print(f"Deleted previous model checkpoint: {previous_model_filename}")
    else:
        print(f"No previous checkpoint to delete at: {previous_model_filename}")