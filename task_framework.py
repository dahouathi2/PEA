import argparse
import torch
import torch.nn as nn

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from data_provider.m4 import M4Meta
from models import Autoformer, DLinear, TimeLLM
from utils.timefeatures import time_features
from data_provider.ean_global_channel import generate_standardization_dicts, save_standardization_data, load_standardization_data
import hashlib

import time
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.losses import smape_loss
import os
from torch.utils.data import DataLoader
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore")

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content, test_MS

from data_provider.ean_global_channel import import_true_promo, import_all, check_saved_standardization_data, delete_saved_standardization_data
from google.cloud import bigquery

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Dataset_Promo_ean_global_channel(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='sold_units', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly', scale_path=None, embedding=True, embedding_dimension = 2, ma=None, diff=None):
        self.features = features
        self.target = target
        self.scale = scale
        self.scale_path = scale_path
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path
        self.ma = ma
        self.diff  = diff
        self.embedding_dict = {}

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.embedding_dim = embedding_dimension
        self.embedding = embedding
        self.seasonal_patterns = seasonal_patterns
        self.history_size = 2
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    def moving_average(self, series, window_size):
        smoothed_series = series.rolling(window=window_size).mean()
        return smoothed_series.fillna(series.mean())
    def difference(self, series):
        return series.diff().dropna()
    def generate_combinations(self, n):
        """Generates all unique combinations of binary values for n binary columns"""
        return [[(i >> j) & 1 for j in range(n)] for i in range(2**n)]
    def deterministic_embedding(self, comb):
        """Generates a deterministic embedding based on a hash of the combination"""
        hash_object = hashlib.sha256(str(comb).encode())
        hash_digest = hash_object.digest()
        seed = int.from_bytes(hash_digest[:4], 'little')
        rng = np.random.default_rng(seed)
        return rng.random(self.embedding_dim)

    def preprocess_pipeline(self, data, id='ean_global_channel'):
        """This function is responsible for all the preprocessing """
        data = data.rename(columns={'end_date': 'date', id: 'id'})
        data = data.drop(['is_promo', 'sub_axis', 'year', 'month', 'week'], axis=1)
        cols = list(data.columns)
        cols.remove(self.target)
        cols.remove('date')
        data = data[cols + [self.target]]  # organize data to date, variables and last is target we're not using date now
        if self.ma is not None:
            data[self.target] = self.moving_average(data[self.target], self.ma)
            data.to_csv(self.root_path + f'{self.flag}_moveingaverage.csv', index=False)
        if self.diff is not None:
            pass
        binary_columns = [col for col in data.columns if col not in ['price_range', 'seasonality_index', 'id', self.target]]

        unique_combinations = self.generate_combinations(len(binary_columns))
        self.embedding_dict = {tuple(comb): self.deterministic_embedding(comb) for comb in unique_combinations}
        
        def get_embedding(row):
            comb = tuple(row[binary_columns])
            return self.embedding_dict[comb]
        if self.embedding:
            embeddings = data[binary_columns].apply(get_embedding, axis=1)
            embedding_columns = [f'embedding_{i+1}' for i in range(self.embedding_dim)]
            embedding_df = pd.DataFrame(embeddings.tolist(), columns=embedding_columns)
            data = pd.concat([data, embedding_df], axis=1)
            data = data.drop(columns=binary_columns)
        
        if self.scale:
            columns_to_standarize = ['price_range', 'sold_units', 'seasonality_index']
            if not check_saved_standardization_data(self.scale_path):
                mean_dict, std_dict, ids = generate_standardization_dicts(data)
                save_standardization_data(mean_dict, std_dict, ids, self.scale_path)
                print(f"standarization dictionaries are created in{self.scale_path}")
            print(f"scaling the data of {self.flag}")
            mean_dict, std_dict, ids = load_standardization_data(self.scale_path)
            standardized_data = pd.DataFrame()
            for id_value, group in data.groupby('id'):
                if id_value in mean_dict:
                    means = pd.Series(mean_dict[id_value])
                    stds = pd.Series(std_dict[id_value])
                    standardized_group = group.copy()
                    for col in columns_to_standarize:
                        if col in means and col in stds:
                            # Standardize each column in the group using the training set stats
                            standardized_group[col] = (group[col] - means[col]) / stds[col]
                        else:
                            print(f"No training data statistics for column: {col} in id: {id_value}. Skipping standardization for this column.")
                    standardized_group['id'] = id_value  # Add id column back
                    standardized_data = pd.concat([standardized_data, standardized_group])
                else:
                    print(f"No training data statistics for id: {id_value}. Skipping standardization for this id.")
            print(f"standarization is over of {self.flag}")
        else:
            standardized_data = data.copy()
        cols = list(standardized_data.columns)
        cols.remove(self.target)
        standardized_data = standardized_data[cols + [self.target]]
        
        return standardized_data
        
    def __read_data__(self):
        if self.flag == 'train':
            dataset = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path)) 
        else:
            dataset = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path.replace('train', 'test')))
        # Preprocessing dataset:
        df = self.preprocess_pipeline(dataset)
        self.ids = df['id'].unique()[:4]
        self.timeseries = [df[df['id']==self.ids[i]].drop('id', axis=1).values for i in range(len(self.ids))]
        self.n_var = self.timeseries[0].shape[1]
    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, self.n_var))
        insample_mask = np.zeros((self.seq_len, self.n_var))
        outsample = np.zeros((self.pred_len + self.label_len, self.n_var))
        outsample_mask = np.zeros((self.pred_len + self.label_len, self.n_var))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        # cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
        #                               high=len(sampled_timeseries),
        #                               size=1)[0]
        if self.flag=='train':
            if self.seq_len <=len(sampled_timeseries)-self.pred_len+1:
                cut_point = np.random.randint(low=self.seq_len,
                                      high=len(sampled_timeseries)-self.pred_len+1,
                                      size=1)[0]
            else:
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries)- self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]
        else:
            cut_point = np.random.randint(low=max(1, len(sampled_timeseries)- self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]
            # if self.flag =='train':
            #     print(cut_point)
        # cut_point = np.random.randint(low=self.seq_len,
        #                               high=len(sampled_timeseries),
        #                               size=1)[0]
        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):] = insample_window
        insample_mask[-len(insample_window):] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window)] = outsample_window
        outsample_mask[:len(outsample_window)] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len, self.n_var))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len, self.n_var))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    drop_last = False
    data_set = Dataset_Promo_ean_global_channel(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=args.scale,
        scale_path=args.scale_path,
        embedding=args.embedding,
        embedding_dimension=args.embedding_dimension
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    print("Shape of X eval", x.shape)
    
    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        
        with autocast():
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = model(
                    x[id_list[i]:id_list[i + 1]],
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                )
        
        # print("Shape of output eval before choosing", outputs.shape)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.tensor(y, dtype=torch.float32).to(accelerator.device)
        true = true[:, -args.pred_len:, f_dim:]
        # print("Shape of y eval", true.shape)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)

        loss = criterion(pred, true)

    model.train()
    return loss

def train_model(model, train_loader, vali_loader, criterion, model_optim, path, args, accelerator):
    if not os.path.exists(path):
        os.makedirs(path)

    args.content = load_content(args)
    time_now = time.time()

    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, verbose=True)


    train_data, train_loader = data_provider(args, 'train')
    train_steps = len(train_loader)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    vali_loader, model, model_optim, scheduler = accelerator.prepare(
            vali_loader, model, model_optim, scheduler)

    for epoch in range(args.train_epochs):
        train_data, train_loader = data_provider(args, 'train')
        train_loader = accelerator.prepare(train_loader)
        
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            with autocast():
                outputs = model(batch_x, None, dec_inp, None)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            batch_y_mark = batch_y_mark[:, -args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                )
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            accelerator.backward(loss)
            model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        accelerator.print('########################################################################')
        vali_loss = test(args, accelerator, model, train_loader, vali_loader, criterion)
        test_loss = vali_loss
        accelerator.print(
            "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, model, path)  # model saving
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    best_model_path = os.path.join(path, 'checkpoint')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), best_model_path)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

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
parser.add_argument('--base_dir', type=str, default='', help=" gcs storage location (bucket link)")
parser.add_argument('--embedding', action='store_true', help='Do the embedding')
parser.add_argument('--embedding_dimension', type=int,default=2, help='dimension of static embedding')
parser.add_argument('--pretrain', action='store_true', help='True then we load the pretrained model')
parser.add_argument('--ma', type=int, default=None, help='Month to do the split train/test')
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

##############################################Initialize ACCELERATOR #############################
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

##################################################################################################
bq_client = bigquery.Client(
    project="itg-bpma-gbl-ww-np",  # GCP project used for running the queries and billing
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

base_dir+= f"{args.channel}Channel_Month{args.month}_{args.num_weeks}Weeks_{args.sequence_n}_seq_n"
if args.fill_discontinuity:
    base_dir += "_filldiscont"
if args.keep_non_promo:
    base_dir += "_keepnonpromo"
if args.scale:
    base_dir+="_scaled"
if args.embedding:
    base_dir+=f"_embedding_{args.embedding_dimension}"
base_dir += '/'+setting


# Saving train and test sets to GCS using gcsfuse
gs_prefix = 'gs://'
gcsfuse_prefix = './gcs/'
if args.base_dir.startswith(gs_prefix):
    args.base_dir = args.base_dir.replace(gs_prefix, gcsfuse_prefix)
dirpath = os.path.split(args.base_dir)[0] + base_dir
if not os.path.isdir(dirpath):
    os.makedirs(dirpath)

train_path = os.path.join(dirpath, "train.csv")
test_path = os.path.join(dirpath, "test.csv")

train_set.to_csv(train_path, index=False)
test_set.to_csv(test_path, index=False)

print(f"Train set saved to: {train_path}")
print(f"Test set saved to: {test_path}")
if args.scale:
     
    args.scale_path = os.path.split(args.base_dir)[0] + 'scale_path/' + base_dir[8:]
    if check_saved_standardization_data(args.scale_path):
        delete_saved_standardization_data(args.scale_path)


########################################################### configuration ####################
args.pred_len = pred_len
args.label_len = pred_len
args.seq_len = int(args.sequence_n*pred_len)
args.root_path = dirpath
args.data_path = 'train.csv'
##############################################################################################
print(f"{args.seq_len}")


    
    

model = TimeLLM.Model(args).float()
model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()
path = os.path.join(os.path.split(args.base_dir)[0] + args.checkpoints,
                    base_dir[8:])  # unique checkpoint saving path
args.content = load_content(args)
if not os.path.exists(path) and accelerator.is_local_main_process:
    os.makedirs(path)


train_data, train_loader = data_provider(args, 'train')
test_data, test_loader = data_provider(args, 'test')

train_model(model, train_loader, test_loader, criterion, model_optim, path, args, accelerator)
