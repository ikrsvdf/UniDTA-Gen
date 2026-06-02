import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors
import math
import pickle
import gzip
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SAScoreCalculator:
    def __init__(self, fpscores_path=None):
        if fpscores_path is None:
            self.fpscores_path = "fpscores.pkl.gz"
        elif fpscores_path.endswith('.pkl.gz'):
            self.fpscores_path = fpscores_path
        else:
            self.fpscores_path = f"{fpscores_path}.pkl.gz"
        self._fscores = None
    
    def _load_fragment_scores(self):
        if self._fscores is not None:
            return
        if not os.path.exists(self.fpscores_path):
            filename = os.path.basename(self.fpscores_path)
            if os.path.exists(filename):
                self.fpscores_path = filename
            else:
                raise FileNotFoundError(f"File {self.fpscores_path} not found.")
        try:
            with gzip.open(self.fpscores_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise IOError(f"读取失败 {self.fpscores_path}: {str(e)}")
        out_dict = {}
        for row in data:
            for j in range(1, len(row)):
                out_dict[row[j]] = float(row[0])
        self._fscores = out_dict
    
    def calculate(self, mol):
        if mol is None:
            return 10.0
        self._load_fragment_scores()
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0
        for bit_id, count in fps.items():
            nf += count
            score1 += self._fscores.get(bit_id, -4) * count
        if nf > 0:
            score1 /= nf
        n_atoms = mol.GetNumAtoms()
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        n_macrocycles = 0
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) > 8:
                n_macrocycles += 1
        size_penalty = n_atoms**1.005 - n_atoms
        stereo_penalty = math.log10(n_chiral + 1)
        spiro_penalty = math.log10(n_spiro + 1)
        bridge_penalty = math.log10(n_bridgehead + 1)
        macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0
        score2 = 0.0 - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty
        score3 = 0.0
        if n_atoms > len(fps):
            score3 = math.log(float(n_atoms) / len(fps)) * 0.5
        raw_score = score1 + score2 + score3
        min_val, max_val = -4.0, 2.5
        sascore = 11.0 - (raw_score - min_val + 1) / (max_val - min_val) * 9.0
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        sascore = max(1.0, min(10.0, sascore))
        return sascore

def compute_chem_props(smiles, sa_calculator):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 0.0
    try:
        qed = QED.qed(mol)
    except:
        qed = 0.0
    try:
        logp = Crippen.MolLogP(mol)
    except:
        logp = 0.0
    try:
        sas = sa_calculator.calculate(mol)
    except:
        sas = 0.0
    return qed, logp, sas

def process_csv_file(input_csv, output_csv, fpscores_path=None, batch_size=1000):
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} does not exist!")
        return
    df = pd.read_csv(input_csv)
    if 'Drug' not in df.columns:
        print("Error: The DataFrame does not have a 'Drug' column!")
        return
    try:
        sa_calculator = SAScoreCalculator(fpscores_path)
    except Exception as e:
        print(f"Failed to initialize SA Score calculator: {e}")
        return
    df['qed'] = 0.0
    df['logp'] = 0.0
    df['sas'] = 0.0
    total_rows = len(df)
    valid_count = 0
    invalid_smiles = []
    for i in tqdm(range(0, total_rows, batch_size), desc="Processing progress"):
        batch = df.iloc[i:i+batch_size]
        for idx, row in batch.iterrows():
            smiles = str(row['Drug']).strip()
            if not smiles or pd.isna(smiles):
                invalid_smiles.append((idx, "Empty SMILES"))
                continue
            qed, logp, sas = compute_chem_props(smiles, sa_calculator)
            df.at[idx, 'qed'] = qed
            df.at[idx, 'logp'] = logp
            df.at[idx, 'sas'] = sas
            if qed > 0 or logp != 0 or sas > 0:
                valid_count += 1
    df.to_csv(output_csv, index=False)
    if invalid_smiles:
        for idx, reason in invalid_smiles[:10]:
            print(f"{idx}: {df.at[idx, 'Drug']} - {reason}")
    return df

if __name__ == "__main__":
    input_file = "/media/ubuntu/disk_1/lcg/DTA/kiba/kiba_dataset_with_3di.csv"
    output_file = "/media/ubuntu/disk_1/lcg/DTA/kiba/kiba_dataset_with_3di.csv"
    fpscores_path = "/media/ubuntu/disk_1/lcg/DTA/data_process/fpscores.pkl.gz"
    if not os.path.exists(fpscores_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(current_dir, "fpscores.pkl.gz")
        if os.path.exists(local_path):
            fpscores_path = local_path
    result_df = process_csv_file(input_file, output_file, fpscores_path)