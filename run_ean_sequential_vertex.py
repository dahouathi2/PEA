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
import pandas

from utils.losses import smape_loss
from utils.m4_summary import M4Summary
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content, test_MS

from data_provider.ean_global_channel import import_true_promo, import_all



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
parser.add_argument('--month', type=str, required=True, help='Month to do the split train/test')
parser.add_argument('--num_weeks', type=int, required=True, help="Minimum number of weeks; must be > 3 * prediction_length")
parser.add_argument('--channel', type=str, choices=[None, 'offline', 'online'], default=None, help="Channel: Both, offline, online")
parser.add_argument('--fill_discontinuity', action='store_true', help='Add the product that has discontinuity in values and interpolate them')
parser.add_argument('--keep_non_promo', action='store_true', help='Keep the products that have no promotions during the whole period')
parser.add_argument('--interpolation', action='store_true', help='Use Full data for long term forecasting')
parser.add_argument('--interpolation_method', action='store_true', help='True then we use PU method for interpolation')

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
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
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
REMOTE_JOB_NAME = "timeseriesllm"
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

os.makedirs(base_dir, exist_ok=True)

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

# Placeholder for saving data
with open(train_path, 'w') as train_file:
    train_file.write(train_set)
with open(test_path, 'w') as test_file:
    test_file.write(test_set)

print(f"Train set saved to: {train_path}")
print(f"Test set saved to: {test_path}")

########################################################### configuration ####################
args.pred_len = pred_len
args.label_len = args.pred_len
args.seq_len = int(2*args.pred_len)
args.root_path = base_dir
args.data_path = 'train.csv'
##############################################################################################




if args.data == 'promo_ean_channel':
    args.seq_len = int(1.75 * args.pred_len)
    args.label_len = args.pred_len
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

    

    

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        base_dir[8:] + setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)



    vertexai.preview.init(remote=False)
    model.train_model.vertex.remote_config.container_uri = "europe-west1-docker.pkg.dev/itg-bpma-gbl-ww-np/yb-vertext-training-rep/yb-vertext-training:latest"
    model.train_model.vertex.remote_config.enable_cuda = True
    model.train_model.vertex.remote_config.accelerator_count = 4
    model.train_model(path)
    torch.save(model.state_dict(), path + '/' + 'checkpoint_v_test1')

    
    train_data, train_loader = data_provider(args, 'train')
    test_data, test_loader = data_provider(args, 'test')

#     best_model_path = path + '/' + 'checkpoint'
#     torch.cuda.synchronize()
#     torch.cuda.empty_cache()
#     unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))

    x, _ = train_loader.dataset.last_insample_window()
    y = test_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    print('########################################################################')

#     model.eval()

#     with torch.no_grad():
#         B, _, C = x.shape
#         dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
#         dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
#         outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
#         id_list = np.arange(0, B, args.eval_batch_size)
#         id_list = np.append(id_list, B)
#         for i in range(len(id_list) - 1):
#             outputs[id_list[i]:id_list[i + 1], :, :] = model(
#                 x[id_list[i]:id_list[i + 1]],
#                 None,
#                 dec_inp[id_list[i]:id_list[i + 1]],
#                 None
#             )
#         accelerator.wait_for_everyone()
#         f_dim = -1 if args.features == 'MS' else 0
#         outputs = outputs[:, -args.pred_len:, f_dim:]
#         outputs = outputs.detach().cpu().numpy()

#         preds = outputs
#         trues = y
#         x = x.detach().cpu().numpy()

#     accelerator.print('test shape:', preds.shape)

#     folder_path = './results/' + args.model + '-' + args.model_comment + '/'
#     if not os.path.exists(folder_path) and accelerator.is_local_main_process:
#         os.makedirs(folder_path)

#     if accelerator.is_local_main_process:
#         forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(args.pred_len)])
#         forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
#         forecasts_df.index.name = 'id'
#         forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
#         forecasts_df.to_csv(folder_path + args.model_id + '_forecast.csv')

#         # calculate metrics
#         accelerator.print(args.model)
#         file_path = folder_path
#         if 'Weekly_forecast.csv' in os.listdir(file_path) \
#                 and 'Monthly_forecast.csv' in os.listdir(file_path) \
#                 and 'Yearly_forecast.csv' in os.listdir(file_path) \
#                 and 'Daily_forecast.csv' in os.listdir(file_path) \
#                 and 'Hourly_forecast.csv' in os.listdir(file_path) \
#                 and 'Quarterly_forecast.csv' in os.listdir(file_path):
#             m4_summary = M4Summary(file_path, args.root_path)
#             # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
#             smape_results, owa_results, mape, mase = m4_summary.evaluate()
#             accelerator.print('smape:', smape_results)
#             accelerator.print('mape:', mape)
#             accelerator.print('mase:', mase)
#             accelerator.print('owa:', owa_results)
#         else:
#             accelerator.print('After all 6 tasks are finished, you can calculate the averaged performance')

# accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     del_files(path)  # delete checkpoint files
#     accelerator.print('success delete checkpoints')
