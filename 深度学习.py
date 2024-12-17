import torch
from torch.utils.data import Dataset
from torch.nn.init import xavier_normal_, constant_
import torch.utils.data as D
from torch import nn
import pandas as pd
import numpy as np
import copy
import os
from tqdm import tqdm  # 修改此处，从标准tqdm导入，而非notebook模式
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
from loguru import logger

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 参数配置
config = {
    "train_path": 'data/used_car_train_20200313.csv',
    "test_path": 'data/used_car_testB_20200421.csv',
    "epoch": 15,
    "batch_size": 512,
    "lr": 0.001,
    "model_ckpt_dir": './',
    "device": 'cuda:0',  # 'cpu'
    "num_cols": ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
                 'v_11', 'v_12', 'v_13', 'v_14'],
    "cate_cols": ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'seller', 'notRepairedDamage']
}

model_config = {
    "is_use_cate_cols": True,
    "embedding_dim": 4,
    "hidden_units": [256, 128, 64, 32]
}
model_config["num_cols"] = config['num_cols']
model_config['cate_cols'] = config['cate_cols']
train_df = pd.read_csv(config['train_path'], sep=' ')
test_df = pd.read_csv(config['test_path'], sep=' ')
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
df.info()
# 连续特征异常值简单处理
df.loc[df['power'] > 600, 'power'] = 600
# 连续特征
for col in config['num_cols']:
    # 绘制密度图
    sns.kdeplot(df[col], fill=True)
    # 设置图形标题和标签
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Density')
    # 保存图片，方便后续查看，若想直接显示图片，可取消注释plt.show()
    plt.savefig(f'{col}_distribution.png')
    # plt.show()

# 离散特征
for col in config['cate_cols']:
    # 统计特征频次
    counts = df[col].value_counts()
    # 绘制条形图
    counts.plot(kind='bar')
    # 设置图形标题和标签
    plt.title(f'{col} Frequencies')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    # 保存图片，方便后续查看，若想直接显示图片，可取消注释plt.show()
    plt.savefig(f'{col}_frequencies.png')
    # plt.show()

# 离散特征编码
vocab_map = defaultdict(dict)
for col in tqdm(config['cate_cols']):
    df[col] = df[col].fillna('-1')
    map_dict = dict(zip(df[col].unique(), range(df[col].nunique())))
    # label enc
    df[col] = df[col].map(map_dict)
    vocab_map[col]['vocab_size'] = len(map_dict)
model_config['vocab_map'] = vocab_map

# 连续特征编码
for col in config['num_cols']:
    df[col] = df[col].fillna(0)
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
train_df = df[df['price'].notna()].reset_index(drop=True)
# 标签范围太大不利于神经网络进行拟合，这里先对其进行log变换
train_df['price'] = np.log(train_df['price'])
test_df = df[df['price'].isna()].reset_index(drop=True)
del test_df['price']


# Dataset构造
class SaleDataset(Dataset):
    def __init__(self, df, cate_cols, num_cols):
        self.df = df
        self.feature_name = cate_cols + num_cols

    def __getitem__(self, index):
        data = dict()
        for col in self.feature_name:
            data[col] = torch.Tensor([self.df[col].iloc[index]]).squeeze(-1)
        if 'price' in self.df.columns:
            data['price'] = torch.Tensor([self.df['price'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.df)


def get_dataloader(df, cate_cols, num_cols, batch_size=256, num_workers=0, shuffle=True):
    dataset = SaleDataset(df, cate_cols, num_cols)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


# sample
train_dataset = SaleDataset(train_df, config['cate_cols'], config['num_cols'])
train_dataset.__getitem__(888)
# user进行Embedding化，有 4个 user，我想把每个user编码成一个8维的向量
num_user = 4
emb_dim = 8
user_emb_layer = nn.Embedding(num_user, emb_dim)
user_emb_layer.weight
query_index = torch.Tensor([0, 2]).long()
user_emb_layer(query_index)
# Embedding层：用于对离散特征进行编码映射
class EmbeddingLayer(nn.Module):
    def __init__(self,
                 vocab_map=None,
                 embedding_dim=None):
        super(EmbeddingLayer, self).__init__()
        self.vocab_map = vocab_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()

        self.emb_feature = []
        # 使用字典来存储每个离散特征的Embedding标
        for col in self.vocab_map.keys():
            self.emb_feature.append(col)
            self.embedding_layer.update({col: nn.Embedding(
                self.vocab_map[col]['vocab_size'],
                self.embedding_dim,
            )})

    def forward(self, X):
        # 对所有的sparse特征挨个进行embedding
        feature_emb_list = []
        for col in self.emb_feature:
            inp = X[col].long().view(-1, 1)
            feature_emb_list.append(self.embedding_layer[col](inp))
        return torch.cat(feature_emb_list, dim=1)


# MLP
class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 final_activation=None,
                 dropout_rates=0,
                 batch_norm=False,
                 use_bias=True):
        super(MLP, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [self.set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if final_activation is not None:
            dense_layers.append(self.set_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers)  # * used to unpack list

    def set_activation(self, activation):
        if isinstance(activation, str):
            if activation.lower() == "relu":
                return nn.ReLU()
            elif activation.lower() == "sigmoid":
                return nn.Sigmoid()
            elif activation.lower() == "tanh":
                return nn.Tanh()
            else:
                return getattr(nn, activation)()
        else:
            return activation

    def forward(self, inputs):
        return self.dnn(inputs)


class SaleModel(nn.Module):
    def __init__(self,
                 is_use_cate_cols=True,
                 vocab_map=None,
                 embedding_dim=16,
                 num_cols=None,
                 cate_cols=None,
                 hidden_units=[256, 128, 64, 32],
                 loss_fun='nn.L1Loss()'):
        super(SaleModel, self).__init__()
        self.is_use_cate_cols = is_use_cate_cols
        self.vocab_map = vocab_map
        self.embedding_dim = embedding_dim
        self.num_cols = num_cols
        self.num_nums_fea = len(num_cols)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)  # self.loss_fun  = nn.L1Loss()

        if is_use_cate_cols:
            self.emb_layer = EmbeddingLayer(vocab_map=vocab_map, embedding_dim=embedding_dim)
            self.mlp = MLP(
                self.num_nums_fea + self.embedding_dim * len(vocab_map),
                output_dim=1,
                hidden_units=self.hidden_units,
                hidden_activations="ReLU",
                final_activation=None,
                dropout_rates=0,
                batch_norm=True,
                use_bias=True)
        else:
            self.mlp = MLP(
                self.num_nums_fea,
                output_dim=1,
                hidden_units=self.hidden_units,
                hidden_activations="ReLU",
                final_activation=None,
                dropout_rates=0,
                batch_norm=True,
                use_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)

    def get_dense_input(self, data):
        dense_input = []
        for col in self.num_cols:
            dense_input.append(data[col])
        return torch.stack(dense_input, dim=-1)

    def forward(self, data):
        dense_fea = self.get_dense_input(data)  # [batch,num_nums_cols]
        if self.is_use_cate_cols:
            sparse_fea = self.emb_layer(data)  # [batch,num_cate_cols,emb]
            sparse_fea = torch.flatten(sparse_fea, start_dim=1)  # [batch,num_cate_cols*emb]
            mlp_input = torch.cat([sparse_fea, dense_fea], axis=-1)  # [batch,num_nums_cols+num_cate_cols*emb]
        else:
            mlp_input = dense_fea

        y_pred = self.mlp(mlp_input)
        # 为了把复杂多变的loss计算对外不感知，所以写在forward里面
        if 'price' in data.keys():
            loss = self.loss_fun(y_pred.squeeze(), data['price'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


# 训练模型，验证模型，这里就是八股文，熟悉基础pipeline
def train_model(model, train_loader, optimizer, device, metric_list=['mean_absolute_error']):
    model.train()
    pred_list = []
    label_list = []
    max_iter = int(train_loader.dataset.__len__() / train_loader.batch_size)
    for idx, data in enumerate(train_loader):
        # 把数据拷贝在指定的device
        for key in data.keys():
            data[key] = data[key].to(device)
        # 模型前向+Loss计算
        output = model(data)
        pred = output['pred']
        loss = output['loss']
        # 八股文完成模型权重更新
        loss.backward()
        optimizer.step()
        model.zero_grad()

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['price'].squeeze(-1).cpu().detach().numpy())

        if idx % 50 == 0:
            logger.info(f"Iter:{idx}/{max_iter} Loss:{round(loss.item(), 4)}")

    res_dict = dict()
    for metric in metric_list:
        res_dict[metric] = eval(metric)(label_list, pred_list)

    return res_dict


def valid_model(model, valid_loader, device, metric_list=['mean_absolute_error']):
    model.eval()
    pred_list = []
    label_list = []

    for data in (valid_loader):
        # 把数据拷贝在指定的device
        for key in data.keys():
            data[key] = data[key].to(device)
        # 模型前向
        output = model(data)
        pred = output['pred']

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['price'].squeeze(-1).cpu().detach().numpy())

    res_dict = dict()
    for metric in metric_list:
        res_dict[metric] = eval(metric)(label_list, pred_list)

    return res_dict


def test_model(model, test_loader, device):
    model.eval()
    pred_list = []

    for data in test_loader:
        # 把数据拷贝在指定的device
        for key in data.keys():
            data[key] = data[key].to(device)
        # 模型前向
        output = model(data)
        pred = output['pred']
        pred_list.extend(pred.squeeze().cpu().detach().numpy())

    return np.array(pred_list)


test_loader = get_dataloader(test_df, config['cate_cols'], config['num_cols'], batch_size=config['batch_size'],
                             num_workers=0, shuffle=False)

n_fold = 7
oof_pre = np.zeros(len(train_df))
y_pre = np.zeros(len(test_df))
device = torch.device(config['device'])

kf = KFold(n_splits=n_fold)
for fold_, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
    logger.info(f"Fold {fold_ + 1}")
    temp_train_df = train_df.iloc[trn_idx].reset_index(drop=True)
    temp_valid_df = train_df.iloc[val_idx].reset_index(drop=True)

    train_loader = get_dataloader(temp_train_df, config['cate_cols'], config['num_cols'],
                                  batch_size=config['batch_size'], num_workers=0, shuffle=True)
    valid_loader = get_dataloader(temp_valid_df, config['cate_cols'], config['num_cols'],
                                  batch_size=config['batch_size'], num_workers=0, shuffle=False)
    # 声明模型
    model = SaleModel(**model_config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # 声明Trainer
    for epoch in range(config['epoch']):
        # 模型训练
        logger.info(f"Start Training Epoch:{epoch + 1}")
        train_metirc = train_model(model, train_loader, optimizer=optimizer, device=device)
        logger.info(f"Train Metric: {train_metirc}")
        # 模型验证
        valid_metric = valid_model(model, valid_loader, device)
        logger.info(f"Valid Metric: {valid_metric}")
    # 保存模型权重和enc_dict
    save_dict = {'model': model.state_dict()}
    torch.save(save_dict, os.path.join(config['model_ckpt_dir'], f'model_{fold_}.pth'))
    # oof推理
    oof_pre[val_idx] = test_model(model, valid_loader, device=device)
    # 测试集推理
    y_pre += np.array(test_model(model, test_loader,device=device))/n_fold
    # 实际价格的预测结果
    oof_pre_ori = np.exp(oof_pre)
    price_ori = np.exp(train_df['price'])
    # 计算均方绝对误差
    mae = mean_absolute_error(price_ori, oof_pre_ori)
    print(f"Out-of-Fold Mean Absolute Error: {mae}")
    res_df = pd.DataFrame()
    res_df['SaleID'] = test_df['SaleID']
    res_df['price'] = np.exp(y_pre)
    res_df.to_csv('torch_baseline.csv', index=False)

