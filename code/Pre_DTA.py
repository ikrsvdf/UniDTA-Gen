import pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import numpy as np
from dgl.dataloading import GraphDataLoader
import os
from utils import DataSet, collate_fn,Tokenizer
from tqdm import tqdm
from model import *
"""kiba/davis/parasite"""



def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)
    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / down)

def get_cindex(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


def calculate_metrics(y_true, y_pred):
    """计算所有指标"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0],
        'Spearman': spearmanr(y_true, y_pred)[0],
        #'Ci': get_cindex(y_true, y_pred),
        'Rm2': get_rm2(y_true, y_pred)
    }
    return metrics
dataset = "parasite"
fold = "fold_0"
start_set = 'warm_start' # drug_coldstart, protein_coldstart,warm_start
grapg_dir = f'./{dataset}/saved_graphs'


project_root =   '/media/ubuntu/disk_1/lcg/DTA'
os.chdir(project_root)
with open(f'{dataset}/{dataset}_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
def test(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    test_losses = []
    test_true = []
    test_pred = []

    test_pbar = tqdm(test_loader, desc=f'test', leave=False)
    with torch.no_grad():
        for batch in test_pbar:
            smiles_seq = batch['smiles_seq'].to(device)
            affinity = batch['affinity'].to(device)
            esm = batch['esm_feat'].to(device)
            di = batch['3di_feat'].to(device)
            qed = batch['qed'].to(device)
            sas = batch['sas'].to(device)
            logp = batch['logp'].to(device)
            batched_graph = batch['graph'].to(device)
            
            mask = create_mask_from_tensor(esm)
            esm = esm.float()
            di = di.float()
            with torch.cuda.amp.autocast(enabled=False):
                outputs, new_drug, lm_loss, kl_loss = model(
                    smiles_seq, esm, di, mask, 
                    affinity, qed, sas, logp, batched_graph  
                )
            loss = criterion(outputs, affinity)
            test_losses.append(loss.item())
            test_true.append(affinity.cpu().numpy())
            test_pred.append(outputs.cpu().numpy())

            test_pbar.set_postfix({'test_loss': f'{loss.item():.4f}'})

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    
    # import pandas as pd
    # results_df = pd.DataFrame({
    #     'True_Value': test_true.flatten(),
    #     'Predicted_Value': test_pred.flatten()
    # })
    # results_df.to_csv('/media/ubuntu/disk_1/lcg/DTA/code/results/parasite/parasite_new_drug_fold_3.csv', index=False)
    
    metrics = calculate_metrics(test_true, test_pred)
    for metric, value in metrics.items():
        tqdm.write(f'{metric}: {value:.4f}')


batch_size  =  64
pdb_dir = None
esm_dir     = f'./{dataset}/esm2'
di_dir      = f'./{dataset}/3di_embeddings'

test_dataset = DataSet(
    f'./{dataset}/data_folds/{start_set}/{fold}/test.csv',
    esm_dir,
    di_dir,
    tokenizer=tokenizer, 
    grapg_dir=grapg_dir,
    dataset =dataset
)

test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn,num_workers=4)

hidden_dim = 128
out_1 = 32
out_2 = 16
grid_size = 2
head = 2
layer_num = 2
pooling = 'avg'
in_node_dim = 92
in_edge_dim = 21
ka_gat_model = KA_GAT(in_node_dim, in_edge_dim, hidden_dim, out_1, out_2, grid_size, head, layer_num, pooling)
ka_gat_model = ka_gat_model.to(device)

model = DrugGenModel(
    tokenizer=tokenizer,
    ka_gat_model=ka_gat_model
).to(device)


pretrained_path = f"./code/results/{dataset}/{start_set}/{fold}/best_Gen_model_mse.pth"
state_dict = torch.load(pretrained_path, map_location='cpu')
model.load_state_dict(state_dict)
test(model, test_loader)
