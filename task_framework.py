from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 50
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        print("######################")
        print(self.description)
        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 50 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous() # B, T, N

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous() # B N T
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


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
def main(args):
    # Ensure all numerical arguments are explicitly cast to their correct types
    try:
        args.is_training = int(args.is_training)
        args.zero_percent = float(args.zero_percent)
        args.sequence_n = int(args.sequence_n)
        args.month = int(args.month)
        args.patch_len = int(args.patch_len)
        args.stride = int(args.stride)
        args.factor = int(args.factor)
        args.enc_in = int(args.enc_in)
        args.dec_in = int(args.dec_in)
        args.n_heads = int(args.n_heads)
        args.c_out = int(args.c_out)
        args.itr = int(args.itr)
        args.d_model = int(args.d_model)
        args.d_ff = int(args.d_ff)
        args.batch_size = int(args.batch_size)
        args.learning_rate = float(args.learning_rate)
        args.llm_layers = int(args.llm_layers)
        args.train_epochs = int(args.train_epochs)
        args.llm_dim = int(args.llm_dim)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Debug print statements to check argument types
    print(f"task_name (type {type(args.task_name)}): {args.task_name}")
    print(f"is_training (type {type(args.is_training)}): {args.is_training}")
    print(f"zero_percent (type {type(args.zero_percent)}): {args.zero_percent}")
    print(f"sequence_n (type {type(args.sequence_n)}): {args.sequence_n}")
    print(f"month (type {type(args.month)}): {args.month}")
    print(f"patch_len (type {type(args.patch_len)}): {args.patch_len}")
    print(f"stride (type {type(args.stride)}): {args.stride}")
    print(f"factor (type {type(args.factor)}): {args.factor}")
    print(f"enc_in (type {type(args.enc_in)}): {args.enc_in}")
    print(f"dec_in (type {type(args.dec_in)}): {args.dec_in}")
    print(f"n_heads (type {type(args.n_heads)}): {args.n_heads}")
    print(f"c_out (type {type(args.c_out)}): {args.c_out}")
    print(f"itr (type {type(args.itr)}): {args.itr}")
    print(f"d_model (type {type(args.d_model)}): {args.d_model}")
    print(f"d_ff (type {type(args.d_ff)}): {args.d_ff}")
    print(f"batch_size (type {type(args.batch_size)}): {args.batch_size}")
    print(f"learning_rate (type {type(args.learning_rate)}): {args.learning_rate}")
    print(f"llm_layers (type {type(args.llm_layers)}): {args.llm_layers}")
    print(f"train_epochs (type {type(args.train_epochs)}): {args.train_epochs}")
    print(f"llm_dim (type {type(args.llm_dim)}): {args.llm_dim}")
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
main(args)
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
args.label_len = args.sequence_n//2 *pred_len
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
args.label_len = args.sequence_n//2 *pred_len
args.seq_len = int(args.sequence_n*pred_len)
args.root_path = dirpath
args.data_path = 'train.csv'
##############################################################################################
print(f"{args.seq_len}")


    
    

model = Model(args).float()
model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()
path = os.path.join(os.path.split(args.base_dir)[0] + args.checkpoints,
                    base_dir[8:])  # unique checkpoint saving path
if not os.path.exists(path) and accelerator.is_local_main_process:
    os.makedirs(path)


train_data, train_loader = data_provider(args, 'train')
test_data, test_loader = data_provider(args, 'test')

train_model(model, train_loader, test_loader, criterion, model_optim, path, args, accelerator)

x, _ = train_loader.dataset.last_insample_window()
y = test_loader.dataset.timeseries
x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
y = torch.tensor(y, dtype=torch.float32).to(accelerator.device)

print('########################################################################')

model.eval()

with torch.no_grad():
    B, _, C = x.shape
    dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
    dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1).float().to(accelerator.device)
    outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
    id_list = np.arange(0, B, args.eval_batch_size)
    id_list = np.append(id_list, B)
    
    with autocast():  # Use autocast for mixed precision
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
    
    if accelerator.distributed_type == "MULTI_GPU":
        accelerator.wait_for_everyone()
    
    f_dim = -1 if args.features == 'MS' else 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    outputs = outputs.detach().cpu().numpy()

    preds = outputs
    trues = np.array(y.cpu())[:, -args.pred_len:, f_dim:]
    x = x.detach().cpu().numpy()

if accelerator.distributed_type == "MULTI_GPU":
    accelerator.wait_for_everyone()
    
accelerator.print('test shape:', preds.shape, trues.shape)

folder_path = os.path.split(args.base_dir)[0] + './results/' + args.model + base_dir[8:] + args.model_comment + '/'
if not os.path.exists(folder_path) and accelerator.is_local_main_process:
    os.makedirs(folder_path)
    print("path created")
print(folder_path)
if accelerator.is_local_main_process:
    ids = test_loader.dataset.ids[:preds.shape[0]]
    forecasts_df = pd.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(args.pred_len)])
    forecasts_df.insert(0, 'id', ids)
    forecasts_df.to_csv(folder_path + args.model_id + '_forecast.csv', index=False)

    # Calculate metrics
    mse_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    smape_list = []
    r2_list = []
    
    for i in range(preds.shape[0]):
        true_values = trues[i].reshape(-1)
        pred_values = preds[i].reshape(-1)
        
        mse = mean_squared_error(true_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, pred_values)
        # Ensure y_true does not contain zeros to avoid division by zero
        mape = mean_absolute_percentage_error(true_values + np.finfo(float).eps, pred_values)
        smape = symmetric_mean_absolute_percentage_error(true_values + np.finfo(float).eps, pred_values)
        r2 = r2_score(true_values, pred_values)
        
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        smape_list.append(smape)
        r2_list.append(r2)
    
    metrics_df = pd.DataFrame({
        'id': ids,
        'MSE': mse_list,
        'RMSE': rmse_list,
        'MAE': mae_list,
        'MAPE': mape_list,
        'sMAPE': smape_list,
        'R2': r2_list
    })
    
    metrics_df.to_csv(folder_path + args.model_id + '_metrics.csv', index=False)