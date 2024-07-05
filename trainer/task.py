import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch CNN Training')
parser.add_argument('--train_uri', dest='train_uri',
                    type=str, help='Storage location of training CSV')
parser.add_argument('--test_uri', dest='test_uri',
                    type=str, help='Storage location of test CSV')
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model directory')
parser.add_argument('--batch_size', dest='batch_size',
                    type=int, default=16, help='Batch size')
parser.add_argument('--epochs', dest='epochs',
                    type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', dest='lr',
                    type=int, default=20, help='Learning rate')
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

class CustomImageDataset(Dataset):
    width = 28
    height = 28

    def __init__(self, data_file, transform=None, target_transform=None):
        self.dataset = pd.read_csv(data_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        label = self.dataset.at[idx, "label"]
        image = self.dataset.iloc[idx,1:]

        # Create a matrix from the pandas.Series
        image = image.to_numpy() * 0.00392156862745098 # 1 / 255
        image = image.reshape(self.width, self.height)
        image = image.astype(float)
        image = torch.Tensor(image)

        if self.target_transform:
            label = self.target_transform(label)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def get_data(train_gcs_uri, test_gcs_uri):

    train_set = CustomImageDataset(train_gcs_uri)
    test_set = CustomImageDataset(test_gcs_uri)

    # HARDCODED batch_size and shuffle-can customize
    batch_size = 64
    shuffle = False

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader

def get_model():
    logging.info("Get model architecture")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_id = "0" if torch.cuda.is_available() else None
    logging.info(f"Device: {device}")

    model = NeuralNetwork()
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, loss, optimizer, device

def train_model(model, loss_func, optimizer, train_loader, test_loader, device):
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        accuracy = 100 * correct
        print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Define a loss function and an optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)

    # Done training
    return model

# import data from Cloud Storage
logging.info('importing training data')
gs_prefix = 'gs://'
gcsfuse_prefix = '/gcs/'

if args.train_uri.startswith(gs_prefix):
    args.train_uri.replace(gs_prefix, gcsfuse_prefix)

if args.test_uri.startswith(gs_prefix):
    args.test_uri.replace(gs_prefix, gcsfuse_prefix)

train_dataset, test_dataset = get_data(train_gcs_uri=args.train_uri,
                                      test_gcs_uri=args.test_uri)

logging.info('starting training')
model, loss, optimizer, device = get_model()
train_model(model, loss, optimizer, train_dataset, test_dataset, device)


# export model to gcs using GCSFuse
logging.info('start saving')
logging.info("Exporting model artifacts ...")
gs_prefix = 'gs://'
gcsfuse_prefix = '/gcs/'
if args.model_dir.startswith(gs_prefix):
    args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
    dirpath = os.path.split(args.model_dir)[0]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

gcs_model_path = os.path.join(os.path.join(args.model_dir, 'model.pth'))
torch.save(model.state_dict(), gcs_model_path)
logging.info(f'Model is saved to {args.model_dir}')
