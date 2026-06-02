
import os, pickle, logging, pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import torch
import numpy as np
import networkx as nx
import math

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import numpy as np
import copy
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling
from dgl.nn.functional import edge_softmax
from jarvis.core.specie import chem_data, get_node_attributes


def calculate_dis(A, B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis


def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)


def encode_bond_14(bond):
    # 7+4+2+2+6 = 21
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1

    bond_type = [0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1

    bond_length = bond_length_approximation(bond.GetBondType())

    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1

    non_bond_feature = [0] * 6

    edge_encode = bond_dir + bond_type + [bond_length, bond_length ** 2] + in_ring + non_bond_feature

    return edge_encode


def non_bonded(charge_list, i, j, dis):
    charge_list = [float(charge) for charge in charge_list]
    q_i = [charge_list[i]]
    q_j = [charge_list[j]]
    q_ij = [charge_list[i] * charge_list[j]]
    dis_1 = [1 / dis]
    dis_2 = [1 / (dis ** 6)]
    dis_3 = [1 / (dis ** 12)]

    return q_i + q_j + q_ij + dis_1 + dis_2 + dis_3



def uff_force_field(mol):
    try:
        mol.RemoveAllConformers()
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol)
        return True
    except Exception as e:
        print(f"UFF: {e}")
        return False

def mmff_force_field(mol):
    try:
        mol.RemoveAllConformers()
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        if mp is None:
            return False
        AllChem.MMFFOptimizeMolecule(mol)
        return True
    except Exception as e:
        print(f"MMFF: {e}")
        return False


def random_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        return True
    except ValueError:
        return False


def check_common_elements(list1, list2, element1, element2):
    for i in range(len(list1)):
        if list1[i] == element1 and list2[i] == element2:
            return True
    return False


def tensor_nan_inf(per_bond_feat):
    nan_exists = any(math.isnan(x) if isinstance(x, float) else False for x in per_bond_feat)
    inf_exists = any(x == float('inf') if isinstance(x, float) else False for x in per_bond_feat)
    ninf_exists = any(x == float('-inf') if isinstance(x, float) else False for x in per_bond_feat)

    if nan_exists or inf_exists or ninf_exists:
        clean_list = [0 if isinstance(x, float) and math.isnan(x) else x for x in per_bond_feat]
        per_bond_feat = [1 if x == float('inf') else -1 if x == float('-inf') else x for x in clean_list]
        return per_bond_feat
    else:
        return per_bond_feat

import torch
import numpy as np
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem

def atom_to_graph(smiles, encoder_atom, encoder_bond, max_atoms=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    else:
        mol = Chem.AddHs(mol)
    
    sps_features = []
    coor = []
    edge_id = []
    atom_charges = []

    smiles_with_hydrogens = Chem.MolToSmiles(mol)

    tmp = []
    for num in smiles_with_hydrogens:
        if num not in ['[', ']', '(', ')']:
            tmp.append(num)

    sm = {}
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        sm[atom_index] = atom.GetSymbol()

    Num_toms = len(tmp)
    if Num_toms > 700:
        return False

    num_conformers = 0
    if mmff_force_field(mol) == True:
        num_conformers = mol.GetNumConformers()
    elif uff_force_field(mol) == True:
        num_conformers = mol.GetNumConformers()
    else:
        Chem.RemoveStereochemistry(mol)
        if mmff_force_field(mol) == True:
            num_conformers = mol.GetNumConformers()
        else: 
            return False  
    
    if num_conformers <= 0:
        return False

    AllChem.ComputeGasteigerCharges(mol)

    for ii, s in enumerate(mol.GetAtoms()):
        per_atom_feat = []
        feat = list(get_node_attributes(s.GetSymbol(), atom_features=encoder_atom))
        per_atom_feat.extend(feat)
        sps_features.append(per_atom_feat)

        pos = mol.GetConformer().GetAtomPosition(ii)
        coor.append([pos.x, pos.y, pos.z])

        charge = s.GetProp("_GasteigerCharge")
        atom_charges.append(charge)

    num_atoms_original = len(sps_features)

    if num_atoms_original > max_atoms:
        sps_features = sps_features[:max_atoms]
        coor = coor[:max_atoms]
        atom_charges = atom_charges[:max_atoms]
        num_atoms_now = max_atoms
    else:
        num_atoms_now = num_atoms_original
    edge_features = []
    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        if src >= max_atoms or dst >= max_atoms:
            continue
        bond_type = bond.GetBondTypeAsDouble()
        src_list.append(src)
        src_list.append(dst)
        dst_list.append(dst)
        dst_list.append(src)
        src_coor = np.array(coor[src])
        dst_coor = np.array(coor[dst])
        s_d_dis = calculate_dis(src_coor, dst_coor)
        per_bond_feat = []
        per_bond_feat.extend(encode_bond_14(bond))
        edge_features.append(per_bond_feat)
        edge_features.append(per_bond_feat)
        edge_id.append([1])
        edge_id.append([1])

    for i in range(len(coor)):
        if i >= max_atoms:
            continue
        coor_i = np.array(coor[i])
        for j in range(i + 1, len(coor)):
            if j >= max_atoms:
                continue
            coor_j = np.array(coor[j])
            s_d_dis = calculate_dis(coor_i, coor_j)
            if 0 < s_d_dis <= 5:
                if check_common_elements(src_list, dst_list, i, j):
                    src_list.extend([i, j])
                    dst_list.extend([j, i])
                    per_bond_feat = [0] * 15
                    per_bond_feat.extend(non_bonded(atom_charges, i, j, s_d_dis))
                    clean_list = tensor_nan_inf(per_bond_feat)
                    edge_features.append(clean_list)
                    edge_features.append(clean_list)
                    edge_id.append([0])
                    edge_id.append([0])

    num_pad = max_atoms - num_atoms_now
    feat_dim = len(sps_features[0]) if sps_features else 0

    if num_pad > 0:
        sps_features += [[0.0]*feat_dim for _ in range(num_pad)]
        coor += [[0.0, 0.0, 0.0] for _ in range(num_pad)] 

    coor_tensor = torch.tensor(coor, dtype=torch.float32)
    edge_feats = torch.tensor(edge_features, dtype=torch.float32)
    edge_id_feats = torch.tensor(edge_id, dtype=torch.float32)
    node_feats = torch.tensor(sps_features, dtype=torch.float32)

    g = dgl.DGLGraph()
    g.add_nodes(max_atoms)  
    if len(src_list) > 0:
        g.add_edges(src_list, dst_list)

    g.ndata['feat'] = node_feats
    g.ndata['coor'] = coor_tensor
    g.edata['feat'] = edge_feats
    g.edata['id'] = edge_id_feats

    return g

def path_complex_mol(Smile, encoder_atom, encoder_bond):
    g = atom_to_graph(Smile, encoder_atom, encoder_bond)
    
    if g != False:
        return g
    else:
        return False
project_root = '/media/ubuntu/disk_1/lcg/DTA' 
os.chdir(project_root)
SAVE_DIR = './parasite/saved_graphs'                  
os.makedirs(SAVE_DIR, exist_ok=True)
encoder_atom = "cgcnn"
encoder_bond = "dim_14"
def build_and_save(df_path: str, split_name: str):
    df = pd.read_csv(df_path)
    smiles_list = df['generated_molecules'].dropna().unique()

    cache = {}                            
    for smi in tqdm(smiles_list, desc=f'Building {split_name}'):
        if smi in cache:                    
            continue
        g = path_complex_mol(smi, encoder_atom, encoder_bond)
        if g is False:                    
            g = None
            print(f'Failed to build graph for SMILES: {smi}')
        cache[smi] = g

    save_path = os.path.join(SAVE_DIR, f'{split_name}_graphs_new_3.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(cache, f)
    logging.info(f'{split_name} graphs saved -> {save_path}  ({len(cache)} unique)')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build_and_save('/media/ubuntu/disk_1/lcg/DTA/parasite/data_folds/warm_start/fold_3/fold_3.csv', 'parasite')
