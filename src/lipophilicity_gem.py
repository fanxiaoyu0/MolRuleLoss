import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm, trange
from copy import deepcopy
import matplotlib.pyplot as plt
from line_profiler import profile
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from GeoGNN import GeoGNN, GEM_Dataset, collate_fn
from utils import archive_code, load_pickle_file, save_pickle_file, load_json_file, save_json_file, get_current_file_name, \
    set_random_seed, get_dataset_meta_data_dict

def construct_data_list():
    df = pd.read_csv(f"../data/intermediate/lipophilicity_deepchem_scaffold_zscore.csv")
    smiles_to_molecule_meta_data_dict = load_pickle_file(f'../data/intermediate/smiles_to_molecule_meta_data_dict_lipophilicity.pkl')
    data_list = []
    for index, row in df.iterrows():
        smiles = row["smiles"]
        if smiles in smiles_to_molecule_meta_data_dict:
            data_item = {
                "smiles": smiles,
                "molecule_meta_data": deepcopy(smiles_to_molecule_meta_data_dict[smiles]),
                "label": row['label_zscore'],
                "dataset_type": row["dataset_type"],
            }
            data_list.append(data_item)
    save_pickle_file(f'../data/intermediate/data_list_lipophilicity_deepchem_scaffold_gem.pkl', data_list)

def get_data_loader():
    batch_size = 32
    data_list = pkl.load(open(f'../data/intermediate/data_list_lipophilicity_deepchem_scaffold_gem.pkl', 'rb'))
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
    data_loader_train = DataLoader(GEM_Dataset(data_list_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_validate = DataLoader(GEM_Dataset(data_list_validate), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(GEM_Dataset(data_list_test), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader_train, data_loader_validate, data_loader_test

class GEM_Regressor(nn.Module):
    def __init__(self):
        super(GEM_Regressor, self).__init__()
        self.encoder = GeoGNN(batch_size=32)
        self.encoder.load_state_dict(torch.load("../weight/regression.pth", weights_only=True))
        self.mlp = nn.Sequential(
            # nn.Linear(32, 32),
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.Linear(128, 128),
            # nn.GELU(),
            # nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            # nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Linear(32, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            # nn.ReLU(),
            nn.Linear(32, 32),
            nn.GELU(),
            # nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, graph_batch):
        node_representation, edge_representation, graph_representaion = self.encoder(graph_batch)
        x = self.mlp(graph_representaion)
        return x

def evaluate(model: GEM_Regressor, data_loader: DataLoader, return_predict_label=False):
    model.eval()
    label_predict_list = torch.tensor([], dtype=torch.float32).cuda()
    label_true_list = torch.tensor([], dtype=torch.float32).cuda()
    smiles_list = []
    with torch.no_grad():
        for data_batch in data_loader:
            graph_batch, label_true_batch, smiles_batch = data_batch
            label_predict_batch = model(graph_batch)
            label_predict_list = torch.cat((label_predict_list, label_predict_batch.detach()), dim=0)
            label_true_list = torch.cat((label_true_list, label_true_batch.detach()), dim=0)
            smiles_list.extend(smiles_batch)
    
    dataset_meta_data_dict = get_dataset_meta_data_dict(file_name="lipophilicity_deepchem_scaffold")
    label_mean = dataset_meta_data_dict["label"]["mean"]
    label_std = dataset_meta_data_dict["label"]["std"]
    label_true_list = label_true_list.squeeze().cpu().numpy() * label_std + label_mean
    label_predict_list = label_predict_list.squeeze().cpu().numpy() * label_std + label_mean

    rmse = round(float(mean_squared_error(label_true_list, label_predict_list) ** 0.5), 3)
    mae = round(float(mean_absolute_error(label_true_list, label_predict_list)), 3)
    r2 = round(float(r2_score(label_true_list, label_predict_list)), 3)
    metric = {'rmse': rmse, 'mae': mae, 'r2': r2}
    if return_predict_label:
        return metric, label_true_list, label_predict_list, smiles_list
    else:
        return metric

def train(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = get_data_loader()

    model = GEM_Regressor()
    model.cuda()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)

    current_best_metric = None
    current_best_epoch = 0
    current_best_weight = None
    metric_list_validate = []
    metric_list_test = []

    for epoch in range(1000):
        model.train()
        for data_batch in data_loader_train:
            graph_batch, label_true_batch, smiles_batch = data_batch
            label_predict_batch = model(graph_batch)
            loss: torch.Tensor = criterion(label_predict_batch, label_true_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        metric_validate = evaluate(model, data_loader_validate)
        metric_test = evaluate(model, data_loader_test)
        metric_list_validate.append(metric_validate)
        metric_list_test.append(metric_test)

        major_metric_name = 'rmse'
        if epoch == 0 or metric_validate[major_metric_name] < current_best_metric[major_metric_name]:
            current_best_metric = metric_validate
            current_best_epoch = epoch
            current_best_weight = deepcopy(model.state_dict())
        print("=========================================================")
        print("epoch", epoch)
        print("metric_validate", metric_validate)
        print("metric_test", metric_test)
        print('current_best_epoch', current_best_epoch)
        print('current_best_metric', current_best_metric)
        print("=========================================================")
    
    torch.save(current_best_weight, f'../weight/{trial_version}.pth')
    metric_list = {
        'metric_list_validate': metric_list_validate, 
        'metric_list_test': metric_list_test
    }
    save_json_file(metric_list, f'../data/seldom/{trial_version}_metric_list.json')

def test(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = get_data_loader()
    model = GEM_Regressor()
    model.cuda()
    model.load_state_dict(torch.load(f"../weight/{trial_version}.pth", weights_only=True))

    metric_train, label_true_train, label_predict_train, smiles_list_train = evaluate(model, data_loader_train, return_predict_label=True)
    metric_validate, label_true_validate, label_predict_validate, smiles_list_validate = evaluate(model, data_loader_validate, return_predict_label=True)
    metric_test, label_true_test, label_predict_test, smiles_list_test = evaluate(model, data_loader_test, return_predict_label=True)

    print("=========================================================")
    print("metric_train", metric_train)
    print("metric_validate", metric_validate)
    print("metric_test", metric_test)
    print("=========================================================")

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
            label_true = float(label_true_list[i])
            label_predict = float(label_predict_list[i])
            error = abs(label_true - label_predict)
            smiles_to_label_and_error_dict[smiles] = {
                'label_true': label_true,
                'label_predict': label_predict,
                'error': error
            }
        smiles_to_label_and_error_dict = {k: v for k, v in sorted(smiles_to_label_and_error_dict.items(), key=lambda item: item[1]['error'], reverse=True)}
        all_dataset_type_dict[dataset_type] = smiles_to_label_and_error_dict
    save_json_file(all_dataset_type_dict, f'../data/result/{trial_version}_smiles_to_label_and_error_dict.json')

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

if __name__ == '__main__':
    trial_version = '16'
    archive_code(trial_version=trial_version)
    set_random_seed(random_seed=1024)
    # construct_data_list()
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
