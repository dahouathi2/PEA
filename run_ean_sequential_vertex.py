import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler

from data_provider.m4 import M4Meta
from models import Autoformer, DLinear
from models import TimeLLM_vertex as TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import pandas as pd

from utils.losses import smape_loss
from utils.m4_summary import M4Summary
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content, test_MS

from data_provider.ean_global_channel import import_true_promo, import_all, check_saved_standardization_data, delete_saved_standardization_data
from google.cloud import bigquery

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100




parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=0, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# data preparation

parser.add_argument('--zero_percent', type=float, required=True, help='Percentage of sales values that are zero')
parser.add_argument('--month', type=int, required=True, help='Month to do the split train/test')
parser.add_argument('--num_weeks', type=int, help="Minimum number of weeks; must be > 3 * prediction_length")
parser.add_argument('--channel', type=str, choices=[None, 'Offline', 'Online'], default=None, help="Channel: Both, offline, online")
parser.add_argument('--fill_discontinuity', action='store_true', help='Add the product that has discontinuity in values and interpolate them')
parser.add_argument('--keep_non_promo', action='store_true', help='Keep the products that have no promotions during the whole period')
parser.add_argument('--interpolation', action='store_true', help='Use Full data for long term forecasting')
parser.add_argument('--interpolation_method', action='store_true', help='True then we use PU method for interpolation')
parser.add_argument('--scale', action='store_true', help='True then we scale')
parser.add_argument('--scale_path', type=str, default='', help=" scale path")
parser.add_argument('--embedding', action='store_true', help='Do the embedding')
parser.add_argument('--embedding_dimension', type=int,default=2, help='dimension of static embedding')
parser.add_argument('--pretrain', action='store_true', help='True then we load the pretrained model')
parser.add_argument('--ma', type=int, help='Month to do the split train/test')
parser.add_argument('--sequence_n', type=int, help='Month to do the split train/test')


# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=3, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

PROJECT_ID = "itg-bpma-gbl-ww-np"  # @param {type:"string"}
REGION = "europe-west1" 
BUCKET_URI = f"gs://your-bucket-name-{PROJECT_ID}-unique"  # @param {type:"string"}
import vertexai
REMOTE_JOB_NAME = "timeseriesllm1"
REMOTE_JOB_BUCKET = f"{BUCKET_URI}/{REMOTE_JOB_NAME}"
##################################################################################################
vertexai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=REMOTE_JOB_BUCKET,
)

##################################################################################################
bq_client = bigquery.Client(
    project=PROJECT_ID,  # GCP project used for running the queries and billing
)

#################################################################################################
print("Looking for number of weeks to take")
_,_,_,pred_len = import_true_promo(
        client=bq_client,
        zero_percent=0,
        month=args.month,
        num_weeks=0,
        channel=args.channel,
        fill_discontinuity=args.fill_discontinuity,
        keep_non_promo=args.keep_non_promo
    )
print(f"{args.seq_len}")

print("Ended up with ", (2+args.sequence_n)*pred_len)
args.num_weeks=(2+args.sequence_n)*pred_len
args.pred_len = pred_len
args.label_len = pred_len
args.seq_len = args.sequence_n*pred_len
print(f"{args.seq_len}")
print("Let's Load the Data")
if args.interpolation:
    final_data, train_set, test_set, pred_len = import_all(
        client=bq_client,
        zero_percent=args.zero_percent,
        month=args.month,
        num_weeks=args.num_weeks,
        channel=args.channel,
        fill_discontinuity=args.fill_discontinuity,
        keep_non_promo=args.keep_non_promo,
        interpolation_method=args.interpolation_method
    )
else :
    final_data, train_set, test_set, pred_len = import_true_promo(
        client=bq_client,
        zero_percent=args.zero_percent,
        month=args.month,
        num_weeks=args.num_weeks,
        channel=args.channel,
        fill_discontinuity=args.fill_discontinuity,
        keep_non_promo=args.keep_non_promo
    )

setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des)
################## 
# Construct the path
base_dir = f"dataset/"
if args.interpolation:
    base_dir += f"interpolation_{args.interpolation_method}/"
else : 
    base_dir += f"true_promo/"

base_dir+= f"{args.channel}Channel_Month{args.month}_{args.num_weeks}Weeks"
if args.fill_discontinuity:
    base_dir += "_filldiscont"
if args.keep_non_promo:
    base_dir += "_keepnonpromo"
if args.scale:
    base_dir+="_scaled"
if args.embedding:
    base_dir+=f"_embedding_{args.embedding_dimension}"
base_dir += '/'+setting
os.makedirs(base_dir, exist_ok=True)

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")


train_set.to_csv(train_path, index=False)
test_set.to_csv(test_path, index=False)

print(f"Train set saved to: {train_path}")
print(f"Test set saved to: {test_path}")
if args.scale:
     
    args.scale_path = 'scale_path/' + base_dir[8:]
    if check_saved_standardization_data(args.scale_path):
        delete_saved_standardization_data(args.scale_path)


########################################################### configuration ####################
args.pred_len = pred_len
args.label_len = pred_len
args.seq_len = int(args.sequence_n*pred_len)
args.root_path = base_dir
args.data_path = 'train.csv'
##############################################################################################
print(f"{args.seq_len}")

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    
    
    
    

    model = TimeLLM.Model(args).float()

    if args.pretrain:
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        'pretrain',
        args.model,
        'pretrain',
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)
        pretrain_path = os.path.join(args.checkpoints,
                        'pretrain/' + setting+'/checkpoint_pretrain')
        if not os.path.exists(pretrain_path):
            raise "can't find pretrained path"
        model.load_state_dict(torch.load(pretrain_path, map_location=accelerator.device), strict=False)

    path = os.path.join(args.checkpoints,
                        base_dir[8:] + '_' + str(ii) + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path):
        os.makedirs(path)


    train_data, train_loader = data_provider(args, 'train')
    test_data, test_loader = data_provider(args, 'test')

    vertexai.preview.init(remote=False)
    model.train_model.vertex.remote_config.container_uri = "europe-west1-docker.pkg.dev/itg-bpma-gbl-ww-np/timeseriesforecasting/torch-train:latest"
    model.train_model.vertex.remote_config.enable_cuda = True
    model.train_model.vertex.remote_config.accelerator_count = 4
    model.train_model(train_loader, test_loader, test_loader,path)
    torch.save(model.state_dict(), path + '/' + 'checkpoint')

    path = './checkpoints'  # unique checkpoint saving path
    #del_files(path)  # delete checkpoint files
    accelerator.print('success delete checkpoints')