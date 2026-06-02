import dgl
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import re
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os, pickle, logging
 

def _load_graph_dict(grapg_dir,split_name: str):
    pkl = os.path.join(grapg_dir, f'{split_name}_graphs.pkl')
    if not os.path.exists(pkl):
        raise FileNotFoundError(f'Pre-built graph file {pkl} not found. run build_save_graphs.py')
    with open(pkl, 'rb') as f:
        return pickle.load(f)
_GRAPH_DICT = {}

def knn(smiles: str, split_name: str ,grapg_dir:str):
    global _GRAPH_DICT
    if split_name not in _GRAPH_DICT:          
        _GRAPH_DICT[split_name] = _load_graph_dict(grapg_dir,split_name)
    g = _GRAPH_DICT[split_name].get(smiles, None)
    if g is None:
        logging.warning(f'Graph missing for {smiles}')
    return g

class DataSet(Dataset):
    def __init__(self, csv_file, esm_dir, di_dir, tokenizer,grapg_dir,dataset, max_smiles_len=100):
        self.data = pd.read_csv(csv_file)
        self.esm_dir  = esm_dir
        self.di_dir   = di_dir
        self.max_smiles_len = max_smiles_len 
        self.tokenizer = tokenizer  
        self.grapg_dir = grapg_dir
        self.dataset = dataset

    def smiles_to_seq(self, smiles):
        #print("Original SMILES:", smiles)
        tokens = self.tokenizer.parse(smiles)  
        if len(tokens) > self.max_smiles_len:
            tokens = tokens[:self.max_smiles_len]
        else:
            tokens = tokens + [2] * (self.max_smiles_len - len(tokens)) 
        return tokens
    

    def _load_esm_3di_features(self, target_key):
        esm_path = os.path.join(self.esm_dir, f"{target_key}.pt")
        di_path  = os.path.join(self.di_dir, f"{target_key}.pt")
        esm = torch.load(esm_path, map_location='cpu')
        di  = torch.load(di_path,  map_location='cpu')
        return esm, di

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = str(row['Drug'])
        target_key = str(row['target_key'])
        affinity = float(row['Y'])
        qed = float(row['qed'])
        logp = float(row['logp'])
        sas = float(row['sas'])

        smiles_seq = self.smiles_to_seq(smiles)  
        esm_feat, di_feat = self._load_esm_3di_features(target_key)
        graph = knn(smiles, self.dataset,grapg_dir=self.grapg_dir)

        return {
            'smiles_seq': torch.tensor(smiles_seq, dtype=torch.long),
            'affinity': torch.tensor(affinity, dtype=torch.float32),
            'qed': torch.tensor(qed, dtype=torch.float32),
            'logp': torch.tensor(logp, dtype=torch.float32),
            'sas': torch.tensor(sas, dtype=torch.float32),
            'esm_feat': esm_feat,
            '3di_feat': di_feat,
            'graph': graph
        }
        
import dgl  # 确保已导入dgl

def collate_fn(batch):
    smiles_seq = torch.stack([item['smiles_seq'] for item in batch])
    affinity = torch.stack([item['affinity'] for item in batch])
    qed = torch.stack([item['qed'] for item in batch])
    logp = torch.stack([item['logp'] for item in batch])
    sas = torch.stack([item['sas'] for item in batch])
    esm_feat = pad_sequence([item['esm_feat'] for item in batch], batch_first=True)
    di_feat = pad_sequence([item['3di_feat'] for item in batch], batch_first=True)
    graphs = [item['graph'] for item in batch]
    batched_graph = dgl.batch(graphs)  
    
    return {
        'smiles_seq': smiles_seq,
        'affinity': affinity,
        'esm_feat': esm_feat,
        '3di_feat': di_feat,
        'qed': qed,
        'sas': sas,
        'logp': logp,
        'graph': batched_graph  
    }
class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>')
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  # saved for future use


    PATTEN = re.compile(r'\[[^\]]+\]'
                        # only some B|C|N|O|P|S|F|Cl|Br|I atoms can omit square brackets
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')

    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()

        for a in tqdm(smiles_set):
            vocabs.update(re.findall(Tokenizer.PATTEN, a))
        return vocabs

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda x: (len(x), x))
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles, return_atom_idx=False):
        l = []
        if return_atom_idx:
            atom_idx=[]
        for i, s in enumerate(('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>')):
            if s not in self.s2i:
                a = 3  
            else:
                a = self.s2i[s]
            l.append(a)
            
            if return_atom_idx and re.fullmatch(Tokenizer.ATOM_PATTEN, s) is not None:
                atom_idx.append(i)
        if return_atom_idx:
            return l, atom_idx
        return l

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()

        smiles = []
        for p in predictions:
            s = []
            for i in p:
                c = self.i2s[i]
                if c == '<eos>':
                    break
                s.append(c)
            smiles.append(''.join(s))

        return smiles


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

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson': pearson_r,
        'RM2': rm2
    }
