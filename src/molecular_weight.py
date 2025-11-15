import numpy as np
import pandas as pd
import json
import pickle as pkl
import random
from tqdm import tqdm, trange
import time
import pdb
import os
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import archive_code, get_current_file_name, set_random_seed

def construct_element_count_vector(smiles):
    element_count_vector = np.zeros(12, dtype=np.float32)
    element_symbol_to_index_dict = {"H": 0, "C": 1, "O": 2, "N": 3, "F": 4, "S": 5, "Cl": 6, "Br": 7, "P": 8, "I": 9, "B": 10, "Si": 11}
    molecule = Chem.MolFromSmiles(smiles)
    molecule = AllChem.AddHs(molecule)
    for atom in molecule.GetAtoms():
        element_symbol = atom.GetSymbol()
        element_count_vector[element_symbol_to_index_dict[element_symbol]] += 1
    return element_count_vector

def construct_smiles_to_element_count_vector_dict():
    df = pd.read_csv(f"../data/raw/chembl_uniform_5.csv")
    label_list = np.array(df["label"])
    print(np.mean(label_list))
    print(np.std(label_list))
    label_list = (label_list - np.mean(label_list)) / np.std(label_list)
    df["label_zscore"] = label_list
    df.to_csv(f"../data/intermediate/chembl_uniform_5_zscore.csv", index=False)

    smiles_list = df["smiles"].tolist()
    element_count_vector_list = []
    for smiles in tqdm(smiles_list):
        element_count_vector = construct_element_count_vector(smiles)
        element_count_vector_list.append(element_count_vector)

    element_count_vector_list = np.array(element_count_vector_list, dtype=np.float32)
    # zscore
    print(np.mean(element_count_vector_list, axis=0))
    print(np.std(element_count_vector_list, axis=0))
    element_count_vector_list = (element_count_vector_list - np.mean(element_count_vector_list, axis=0)) / np.std(element_count_vector_list, axis=0)
    smiles_to_element_count_vector_dict = {}
    for i in range(len(smiles_list)):
        smiles_to_element_count_vector_dict[smiles_list[i]] = element_count_vector_list[i]
    pkl.dump(smiles_to_element_count_vector_dict, open(f'../data/intermediate/smiles_to_element_count_vector_dict.pkl', 'wb'))

def split_dataset_test_by_molecule_weight():
    df = pd.read_csv("../data/intermediate/chembl_uniform_5_zscore.csv")
    df.sort_values(by="label", inplace=True)
    dataset_type_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['label'] > 601:
            dataset_type_list.append("test")
        else:
            if random.random() < 0.25:
                dataset_type_list.append("validate")
            else:
                dataset_type_list.append("train")
    df["dataset_type"] = dataset_type_list
    print(df["dataset_type"].value_counts())
    df.to_csv("../data/intermediate/chembl_uniform_5_tail_test_random_validate.csv", index=False)

def construct_data_list():
    df = pd.read_csv(f"../data/intermediate/chembl_uniform_5_tail_test_random_validate.csv")
    smiles_to_element_count_vector_dict = pkl.load(open(f'../data/intermediate/smiles_to_element_count_vector_dict.pkl', 'rb'))
    data_list = []
    for index, row in df.iterrows():
        smiles = row["smiles"]
        data_item = {
            "smiles": smiles,
            "element_count_vector": smiles_to_element_count_vector_dict[smiles],
            "label": row['label_zscore'],
            "dataset_type": row["dataset_type"],
        }
        data_list.append(data_item)

    data_list_train = []
    data_list_validate = []
    data_list_test = []
    for data_item in data_list:
        if data_item['dataset_type'] == 'train':
            data_list_train.append(data_item)
        elif data_item['dataset_type'] == 'validate':
            data_list_validate.append(data_item)
        elif data_item['dataset_type'] == 'test':
            data_list_test.append(data_item)

    ratio = 0.05
    random.shuffle(data_list_train)
    random.shuffle(data_list_validate)
    data_list_train = data_list_train[:int(len(data_list_train) * ratio)]
    data_list_validate = data_list_validate[:int(len(data_list_validate) * ratio)]
    data_list = data_list_train + data_list_validate + data_list_test
    print(len(data_list_train), len(data_list_validate), len(data_list_test), len(data_list))
    pkl.dump(data_list, open(f'../data/intermediate/data_list_molecular_weight.pkl', 'wb'))


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def get_data_loader():
    batch_size = 256
    data_list = pkl.load(open(f'../data/intermediate/data_list_molecular_weight.pkl', 'rb'))
    data_list_train = []
    data_list_validate = []
    data_list_test = []
    for data_item in data_list:
        if data_item['dataset_type'] == 'train':
            data_list_train.append(data_item)
        elif data_item['dataset_type'] == 'validate':
            data_list_validate.append(data_item)
        elif data_item['dataset_type'] == 'test':
            data_list_test.append(data_item)
    
    
    # print(len(data_list_train), len(data_list_validate), len(data_list_test))
    data_loader_train = DataLoader(MyDataset(data_list_train), batch_size=batch_size, shuffle=True)
    data_loader_validate = DataLoader(MyDataset(data_list_validate), batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(MyDataset(data_list_test), batch_size=batch_size, shuffle=True)
    return data_loader_train, data_loader_validate, data_loader_test

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            # nn.BatchNorm1d(12),
            nn.Linear(12, 12),
            nn.GELU(),
            # nn.ReLU(),
            # nn.ReLU(),
            # nn.Linear(12, 12),
            # nn.GELU(),
            # nn.Sigmoid(),
            # nn.BatchNorm1d(12),
            nn.Linear(12, 1),
        )

        # self.fully_connected_layer_1 = nn.Linear(12, 12)
        # self.relu = nn.ReLU()
        # self.fully_connected_layer_2 = nn.Linear(12, 1)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

        # 将标准矩阵初始化为可学习参数 (12x12)
        # 最佳实践：使用 Xavier/Glorot 初始化（适用于线性层）
        self.standard_matrix = nn.Parameter(
            torch.empty(12, 12),
            requires_grad=True
        )
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        # Xavier/Glorot 初始化（保持输入输出的方差一致）
        nn.init.xavier_uniform_(self.standard_matrix, gain=1.0)
        
        # 或者使用更简单的随机初始化（可选）
        # nn.init.uniform_(self.standard_matrix, -0.1, 0.1)
        
        # 如果希望矩阵初始接近单位矩阵（适合距离度量）
        # nn.init.eye_(self.standard_matrix)
        # self.standard_matrix.data += 0.01 * torch.randn_like(self.standard_matrix)

    def forward(self, element_count_vector):
        return self.mlp(element_count_vector)

        x = self.fully_connected_layer_1(element_count_vector)
        x = x + element_count_vector
        x = self.relu(x)
        x = self.fully_connected_layer_2(x)
        return x

def evaluate(model: MLP, data_loader):
    model.eval()
    label_true_list = []
    label_predict_list = []
    # with torch.no_grad():
    for data_batch in data_loader:
        label_true_batch = data_batch['label'].cuda().to(torch.float32).unsqueeze(1)
        element_count_vector = data_batch['element_count_vector'].cuda().requires_grad_()
        label_predict_batch = model(element_count_vector)

        mean_gradient = torch.mean(gradients(label_predict_batch, element_count_vector), dim=0)
        mean_gradient = torch.div(mean_gradient * 156.01707373005718, torch.tensor([12.354086, 8.779627, 2.5210264, 2.0930214, 1.1193496, 0.6389612, 0.5923029, 0.26856133, 0.16532433, 0.06922156, 0.06374216, 0.08806009]).cuda()).cpu().tolist()
        mean_gradient = [round(item, 2) for item in mean_gradient]
        relative_atom_mass = [1, 12, 16, 14, 19, 32, 35.5, 80, 31, 127, 11, 28]
        # print(mean_gradient)
        # print(relative_atom_mass)

        label_true_list.extend(label_true_batch.cpu().numpy().squeeze().tolist())
        label_predict_list.extend(label_predict_batch.detach().cpu().numpy().squeeze().tolist())

    label_true_list = np.array(label_true_list) * 156.01707373005718 + 430.9959537037038
    label_predict_list = np.array(label_predict_list) * 156.01707373005718 + 430.9959537037038

    rmse = round(float(mean_squared_error(label_true_list, label_predict_list) ** 0.5), 3)
    mae = round(float(mean_absolute_error(label_true_list, label_predict_list)), 3)
    r2 = round(float(r2_score(label_true_list, label_predict_list)), 3)
    metric = {'rmse': rmse, 'mae': mae, 'r2': r2}
    return metric

def plot_metric_evolution_during_training(trial_version, epoch, metric_list_train, metric_list_validate, metric_list_test, learning_rate_list):
    major_metric = 'rmse'
    major_metric_list_train = [metric[major_metric] for metric in metric_list_train]
    major_metric_list_validate = [metric[major_metric] for metric in metric_list_validate]
    major_metric_list_test = [metric[major_metric] for metric in metric_list_test]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(15, 10)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel(major_metric)
    ax1.plot(range(epoch + 1), major_metric_list_train, label='train', color='tab:blue')
    ax1.plot(range(epoch + 1), major_metric_list_validate, label='validate', color='tab:orange')
    ax1.plot(range(epoch + 1), major_metric_list_test, label='test', color='tab:red')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('learning rate')
    ax2.plot(range(epoch + 1), learning_rate_list, label='learning rate', color='tab:green')
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax2.legend(loc='upper right')
    plt.savefig(f'../data/result/{trial_version}_metric_evolution.png')

def gradients(output, input_, order=1):
    if order == 1:
        return torch.autograd.grad(output, input_, grad_outputs=torch.ones_like(output), create_graph=True, only_inputs=True, )[0]
    else:
        return gradients(gradients(output, input_), input_, order=order - 1)

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.relative_atom_mass = torch.tensor([30.97, 126.9, 10.81, 28.09]).cuda()
        self.relative_atom_mass = torch.mul(self.relative_atom_mass, torch.tensor([0.16532433, 0.06922156, 0.06374216, 0.08806009]).cuda()) / 156.01707373005718
        # self.relative_atom_mass = torch.tensor([126.9, 10.81, 28.09]).cuda()
        # self.relative_atom_mass = torch.mul(self.relative_atom_mass, torch.tensor([0.06922156, 0.06374216, 0.08806009]).cuda()) / 156.01707373005718
        # self.relative_atom_mass = torch.tensor([1.008, 12.011, 15.999, 14.007, 18.998, 32.06, 35.45, 79.904, 30.973, 126.9, 10.81, 28.085]).cuda()
        # self.relative_atom_mass = torch.mul(self.relative_atom_mass, torch.tensor([12.354086, 8.779627, 2.5210264, 2.0930214, 1.1193496, 0.6389612, 0.5923029, 0.26856133, 0.16532433, 0.06922156, 0.06374216, 0.08806009]).cuda()) / 156.01707373005718

    def forward(self, input_, label_true, label_predict):
        mse_loss = self.mse_loss(label_predict, label_true)
        gradient = gradients(label_predict, input_)
        gradient = gradient[:, -4:]
        # gradient = gradient[:, -3:]
        gradient_loss = torch.mean(torch.square(gradient - self.relative_atom_mass.unsqueeze(0).repeat(input_.shape[0], 1)))
        # print(mse_loss)
        # print(gradient_loss * 100)
        return mse_loss + gradient_loss * 100

class MyLoss_Pairwise(nn.Module):
    def __init__(self):
        super(MyLoss_Pairwise, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.relative_atom_mass = torch.tensor([30.97, 126.9, 10.81, 28.09]).cuda()
        # self.relative_atom_mass = torch.mul(self.relative_atom_mass, torch.tensor([0.16532433, 0.06922156, 0.06374216, 0.08806009]).cuda()) / 156.01707373005718
        # self.relative_atom_mass = torch.tensor([126.9, 10.81, 28.09]).cuda()
        # self.relative_atom_mass = torch.mul(self.relative_atom_mass, torch.tensor([0.06922156, 0.06374216, 0.08806009]).cuda()) / 156.01707373005718
        
        # self.relative_atom_mass = torch.tensor([1.008, 12.011, 15.999, 14.007, 18.998, 32.06, 35.45, 79.904, 30.973, 126.9, 10.81, 28.085]).cuda()
        # self.relative_atom_mass = torch.mul(self.relative_atom_mass, torch.tensor([12.354086, 8.779627, 2.5210264, 2.0930214, 1.1193496, 0.6389612, 0.5923029, 0.26856133, 0.16532433, 0.06922156, 0.06374216, 0.08806009]).cuda()) / 156.01707373005718
        # self.standard_matrix = self._create_standard_matrix()
        
        # TODO: 将标准矩阵设置为可学习的参数，是一个 12*12 的矩阵

        self.loss_weight = 10

        

    def _create_standard_matrix(self):
        """创建原子质量的标准差值矩阵"""
        # 生成 12x12 的矩阵，每个元素是 mass[i] - mass[j]
        mass = self.relative_atom_mass.unsqueeze(1)  # 转换为列向量
        return mass - mass.T  # 广播计算差值矩阵

    def forward(self, input_, label_true, label_predict, standard_matrix):
        mse_loss = self.mse_loss(label_predict, label_true)
        gradient = gradients(label_predict, input_)
        assert gradient.size(1) == 12, "梯度维度应为12个特征"
        pred_matrix = self._create_prediction_matrix(gradient)
        batch_size = gradient.size(0)
        target_matrix = standard_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        matrix_loss = F.mse_loss(pred_matrix, target_matrix)
        # print("mse_loss", mse_loss)
        # print("matrix_loss", matrix_loss)
        return mse_loss + matrix_loss * self.loss_weight

    def _create_prediction_matrix(self, gradient):
        # gradient shape: (batch_size, 12)
        x = gradient.unsqueeze(2)  # (batch, 12, 1)
        y = gradient.unsqueeze(1)  # (batch, 1, 12)
        return x - y  # 广播得到 (batch, 12, 12)

def train(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = get_data_loader()

    model = MLP()
    model.cuda()
    # criterion = nn.MSELoss()
    # criterion = MyLoss()
    criterion = MyLoss_Pairwise()
    optimizer = optim.Adam(model.parameters(), lr=3e-2, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15) 

    current_best_metric = None
    current_best_epoch = 0
    current_best_weight = None

    metric_list_train = []
    metric_list_validate = []
    metric_list_test = []
    learning_rate_list = []

    for epoch in range(2000):
        model.train()
        for data_batch in data_loader_train:
            label_true_batch = data_batch['label'].cuda().to(torch.float32).unsqueeze(1)
            element_count_vector = data_batch['element_count_vector'].cuda().requires_grad_()
            label_predict_batch = model(element_count_vector)
            mean_gradient = torch.mean(gradients(label_predict_batch, element_count_vector), dim=0)
            mean_gradient = torch.div(mean_gradient * 156.01707373005718, torch.tensor([12.354086, 8.779627, 2.5210264, 2.0930214, 1.1193496, 0.6389612, 0.5923029, 0.26856133, 0.16532433, 0.06922156, 0.06374216, 0.08806009]).cuda()).cpu().tolist()
            mean_gradient = [round(item, 2) for item in mean_gradient]
            relative_atom_mass = [1, 12, 16, 14, 19, 32, 35.5, 80, 31, 127, 11, 28]
            # print(mean_gradient)
            # print(relative_atom_mass)
            # loss = criterion(label_predict_batch, label_true_batch)
            loss = criterion(element_count_vector, label_true_batch, label_predict_batch, model.standard_matrix)
            # print(label_predict_batch)
            # print(label_true_batch)
            # pdb.set_trace()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        metric_train = evaluate(model, data_loader_train)
        metric_validate = evaluate(model, data_loader_validate)
        metric_test = evaluate(model, data_loader_test)
        metric_list_train.append(metric_train)
        metric_list_validate.append(metric_validate)
        metric_list_test.append(metric_test)
        learning_rate_list.append(scheduler.get_last_lr())

        major_metric_name = 'rmse'
        if epoch == 0 or metric_validate[major_metric_name] < current_best_metric[major_metric_name]:
            current_best_metric = metric_validate
            current_best_epoch = epoch
            current_best_weight = deepcopy(model.state_dict())
            # torch.save(model.state_dict(), f'../weight/{trial_version}.pth')
        if epoch % 20 == 0:
            print("=========================================================")
            print("epoch", epoch)
            print("metric_train", metric_train)
            print("metric_validate", metric_validate)
            print("metric_test", metric_test)
            print('current_best_epoch', current_best_epoch)
            print('current_best_metric', current_best_metric)
            print("=========================================================")

            # print("========================== learned ======================")
            # print(model.standard_matrix)
            # print("=========================================================")

            # print("==============================  original ================")
            # relative_atom_mass = torch.tensor([1.008, 12.011, 15.999, 14.007, 18.998, 32.06, 35.45, 79.904, 30.973, 126.9, 10.81, 28.085]).cuda()
            # relative_atom_mass = torch.mul(relative_atom_mass, torch.tensor([12.354086, 8.779627, 2.5210264, 2.0930214, 1.1193496, 0.6389612, 0.5923029, 0.26856133, 0.16532433, 0.06922156, 0.06374216, 0.08806009]).cuda()) / 156.01707373005718
            # mass = relative_atom_mass.unsqueeze(1)  # 转换为列向量
            # print(mass - mass.T)  # 广播计算差值矩阵
            # print("=========================================================")

    p = multiprocessing.Process(target=plot_metric_evolution_during_training, args=(trial_version, epoch, metric_list_train, metric_list_validate, metric_list_test, learning_rate_list))
    p.start()
    torch.save(current_best_weight, f'../weight/{trial_version}.pth')
    
    metric_list = {
        'metric_list_train': metric_list_train, 
        'metric_list_validate': metric_list_validate, 
        'metric_list_test': metric_list_test
    }
    json.dump(metric_list, open(f'../data/seldom/{trial_version}_metric_list.json', 'w'), indent=4)

def get_predict_label(model: MLP, data_loader):
    model.eval()
    label_true_list = []
    label_predict_list = []
    smiles_list = []
    with torch.no_grad():
        for data_batch in data_loader:
            label_true_batch = data_batch['label'].cuda().to(torch.float32).unsqueeze(1)
            element_count_vector = data_batch['element_count_vector'].cuda()
            label_predict_batch = model(element_count_vector)

            label_true_list.extend(label_true_batch.cpu().numpy().squeeze().tolist())
            label_predict_list.extend(label_predict_batch.cpu().numpy().squeeze().tolist())
            smiles_list.extend(data_batch['smiles'])

    label_true_list = np.array(label_true_list) * 156.01707373005718 + 430.9959537037038
    label_predict_list = np.array(label_predict_list) * 156.01707373005718 + 430.9959537037038
    return label_true_list, label_predict_list, smiles_list

def test(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = get_data_loader()

    model = MLP()
    model.load_state_dict(torch.load(f'../weight/{trial_version}.pth'))
    model.cuda()

    metric_train = evaluate(model, data_loader_train)
    metric_validate = evaluate(model, data_loader_validate)
    metric_test = evaluate(model, data_loader_test)

    print("=========================================================")
    print("metric_train", metric_train)
    print("metric_validate", metric_validate)
    print("metric_test", metric_test)
    print("=========================================================")

    label_true_train, label_predict_train, smiles_list_train = get_predict_label(model, data_loader_train)
    label_true_validate, label_predict_validate, smiles_list_validate = get_predict_label(model, data_loader_validate)
    label_true_test, label_predict_test, smiles_list_test = get_predict_label(model, data_loader_test)

    all_dataset_type_dict = {}
    for dataset_type in ['train', 'validate', 'test']:
        if dataset_type == 'train':
            label_true_list, label_predict_list, smiles_list = label_true_train, label_predict_train, smiles_list_train
        elif dataset_type == 'validate':
            label_true_list, label_predict_list, smiles_list = label_true_validate, label_predict_validate, smiles_list_validate
        elif dataset_type == 'test':
            label_true_list, label_predict_list, smiles_list = label_true_test, label_predict_test, smiles_list_test
        
        smiles_to_label_and_error_dict = {}
        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            label_true = label_true_list[i]
            label_predict = label_predict_list[i]
            error = abs(label_true - label_predict)
            smiles_to_label_and_error_dict[smiles] = {
                'label_true': label_true,
                'label_predict': label_predict,
                'error': error
            }
        smiles_to_label_and_error_dict = {k: v for k, v in sorted(smiles_to_label_and_error_dict.items(), key=lambda item: item[1]['error'], reverse=True)}
        all_dataset_type_dict[dataset_type] = smiles_to_label_and_error_dict
    json.dump(all_dataset_type_dict, open(f'../data/result/{trial_version}_smiles_to_label_and_error_dict.json', 'w'), indent=4)

    plt.figure(figsize=(28, 8))
    plt.subplot(1, 3, 1)
    plt.plot([160, 700], [160, 700], label='y=x', color='tab:green')
    plt.scatter(label_true_train, label_predict_train, s=1, label='train', color='tab:blue')
    plt.xlabel('label_true')
    plt.ylabel('label_predict')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot([160, 700], [160, 700], label='y=x', color='tab:green')
    plt.scatter(label_true_validate, label_predict_validate, s=1, label='validate', color='tab:orange')
    plt.xlabel('label_true')
    plt.ylabel('label_predict')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot([160, 700], [160, 700], label='y=x', color='tab:green')
    plt.scatter(label_true_test, label_predict_test, s=1, label='test', color='tab:red')
    plt.xlabel('label_true')
    plt.ylabel('label_predict')
    plt.legend()
    plt.savefig(f'../data/result/{trial_version}_scatter.png')
    plt.close()

    return metric_train, metric_validate, metric_test

def plot_metric_evolution_after_training(trial_version):
    metric_list = json.load(open(f'../data/seldom/{trial_version}_metric_list.json', 'r'))
    metric_list_train, metric_list_validate, metric_list_test = metric_list['metric_list_train'], metric_list['metric_list_validate'], metric_list['metric_list_test']
    major_metric = 'rmse'
    major_metric_list_train = [metric[major_metric] for metric in metric_list_train]
    major_metric_list_validate = [metric[major_metric] for metric in metric_list_validate]
    major_metric_list_test = [metric[major_metric] for metric in metric_list_test]

    plt.figure(figsize=(15, 10))
    plt.xlabel('epoch')
    plt.ylabel(major_metric)
    plt.plot(range(len(major_metric_list_train)), major_metric_list_train, label='train', color='tab:blue')
    plt.plot(range(len(major_metric_list_validate)), major_metric_list_validate, label='validate', color='tab:orange')
    plt.plot(range(len(major_metric_list_test)), major_metric_list_test, label='test', color='tab:red')
    plt.tick_params(axis='y')
    plt.legend(loc='upper left')
    plt.savefig(f'../data/result/{trial_version}_metric_evolution_1.png')

def temp_0():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 元素符号和原子质量对应表 (按您的顺序)
    elements = ['H', 'C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'B', 'Si']
    atomic_masses = [1.008, 12.011, 15.999, 14.007, 18.998, 32.06, 35.45, 79.904, 30.973, 126.9, 10.81, 28.085]

    # 假设已经加载模型
    model = MLP()
    model.load_state_dict(torch.load(f'../weight/molecular_weight_17_1024.pth'))
    model.eval()

    # 获取学习到的矩阵
    learned_matrix = model.standard_matrix.detach().cpu().numpy()
    print(learned_matrix)

    # 计算 groundtruth 矩阵
    relative_atom_mass = torch.tensor(atomic_masses).cuda()
    weights = torch.tensor([12.354086, 8.779627, 2.5210264, 2.0930214, 1.1193496, 
                        0.6389612, 0.5923029, 0.26856133, 0.16532433, 
                        0.06922156, 0.06374216, 0.08806009]).cuda()
    weighted_mass = torch.mul(relative_atom_mass, weights) / 156.01707373005718
    mass_diff_matrix = (weighted_mass.unsqueeze(1) - weighted_mass.unsqueeze(0)).cpu().numpy()
    print(mass_diff_matrix)

    # 创建画布
    plt.figure(figsize=(16, 6), dpi=300)
    # plt.rcParams['font.family'] = 'Arial'  # 学术论文常用字体
    plt.rcParams['font.size'] = 10

    # 子图1: Learned Rules Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(learned_matrix, 
                cmap='coolwarm', 
                center=0,
                annot=True, 
                fmt=".2f",
                xticklabels=elements,
                yticklabels=elements,
                cbar_kws={'label': 'Weight Value'})
    plt.title('Learned Rules Matrix', pad=20, fontweight='bold')
    plt.xlabel('Element', fontweight='bold')
    plt.ylabel('Element', fontweight='bold')

    # 子图2: Groundtruth Rules Matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(mass_diff_matrix, 
                cmap='coolwarm', 
                center=0,
                annot=True, 
                fmt=".2f",
                xticklabels=elements,
                yticklabels=elements,
                cbar_kws={'label': 'Mass Difference'})
    plt.title('Groundtruth Rules Matrix', pad=20, fontweight='bold')
    plt.xlabel('Element', fontweight='bold')

    # 调整布局
    plt.tight_layout(pad=3.0)

    # 保存为高分辨率图片（适合论文投稿）
    plt.savefig('../data/temp/rules_matrix_comparison.png', bbox_inches='tight', transparent=True)
    plt.close()

def temp_1():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 元素符号和原子质量对应表
    elements = ['H', 'C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'B', 'Si']
    atomic_masses = [1.008, 12.011, 15.999, 14.007, 18.998, 32.06, 35.45, 79.904, 30.973, 126.9, 10.81, 28.085]

    # 加载模型和数据
    model = MLP()
    model.load_state_dict(torch.load(f'../weight/molecular_weight_17_1024.pth'))
    model.eval()

    # 获取矩阵数据
    learned_matrix = model.standard_matrix.detach().cpu().numpy()
    
    # 计算groundtruth矩阵
    relative_atom_mass = torch.tensor(atomic_masses).cuda()
    weights = torch.tensor([12.354086, 8.779627, 2.5210264, 2.0930214, 1.1193496, 
                          0.6389612, 0.5923029, 0.26856133, 0.16532433, 
                          0.06922156, 0.06374216, 0.08806009]).cuda()
    weighted_mass = torch.mul(relative_atom_mass, weights) / 156.01707373005718
    mass_diff_matrix = (weighted_mass.unsqueeze(1) - weighted_mass.unsqueeze(0)).cpu().numpy()

    # 计算误差矩阵（绝对误差）
    error_matrix = np.abs(learned_matrix - mass_diff_matrix) * 10 ** 5

    # 创建画布
    plt.figure(figsize=(10, 8), dpi=300)
    plt.rcParams['font.size'] = 10

    # 绘制误差热力图
    ax = sns.heatmap(error_matrix, 
                    cmap='YlOrRd',  # 黄-橙-红色阶，适合显示误差
                    annot=True, 
                    fmt=".2f",
                    xticklabels=elements,
                    yticklabels=elements,
                    cbar_kws={'label': 'Absolute Error'})
    
    plt.title('Absolute Error Between Learned and Groundtruth Matrices (* 10^5)', 
             pad=20, fontweight='bold')
    plt.xlabel('Element', fontweight='bold')
    plt.ylabel('Element', fontweight='bold')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('../data/temp/error_matrix.png', bbox_inches='tight', transparent=True)
    plt.close()

    print("Error heatmap saved to ../data/temp/error_matrix.png")

if __name__ == '__main__':
    # temp_0()
    # temp_1()
    trial_version = '18'
    archive_code(trial_version=trial_version)
    set_random_seed(random_seed=1024)
    # construct_smiles_to_element_count_vector_dict()
    # split_dataset_test_by_molecule_weight()
    construct_data_list()
    for random_seed in range(1024, 1024 + 5):
        set_random_seed(random_seed=random_seed)
        train(trial_version=f'{get_current_file_name()}_{trial_version}_{random_seed}')
        test(trial_version=f'{get_current_file_name()}_{trial_version}_{random_seed}')

    metric_train_list = []
    metric_validate_list = []
    metric_test_list = []
    for random_seed in range(1024, 1024 + 5):
        set_random_seed(random_seed=random_seed)
        metric_train, metric_validate, metric_test = test(trial_version=f'{get_current_file_name()}_{trial_version}_{random_seed}')
        metric_train_list.append(metric_train)
        metric_validate_list.append(metric_validate)
        metric_test_list.append(metric_test)
    print("=========================================================")
    metric_train_mean = {}
    metric_validate_mean = {}
    metric_test_mean = {}
    metric_train_std = {}
    metric_validate_std = {}
    metric_test_std = {}
    for metric_name in ['rmse', 'mae', 'r2']:
        metric_train_mean[metric_name] = round(float(np.mean([metric[metric_name] for metric in metric_train_list])), 3)
        metric_validate_mean[metric_name] = round(float(np.mean([metric[metric_name] for metric in metric_validate_list])), 3)
        metric_test_mean[metric_name] = round(float(np.mean([metric[metric_name] for metric in metric_test_list])), 3)
        metric_train_std[metric_name] = round(float(np.std([metric[metric_name] for metric in metric_train_list])), 3)
        metric_validate_std[metric_name] = round(float(np.std([metric[metric_name] for metric in metric_validate_list])), 3)
        metric_test_std[metric_name] = round(float(np.std([metric[metric_name] for metric in metric_test_list])), 3)
    print("metric_train_mean", metric_train_mean)
    print("metric_validate_mean", metric_validate_mean)
    print("metric_test_mean", metric_test_mean)
    print("=========================================================")
    print("All is well!")
