"""esm2_t33_650M_UR50D"""
import csv
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, T5EncoderModel
import esm
from esm import Alphabet, pretrained
import re
import os

project_root = '/media/ubuntu/disk_1/lcg/DTA'
os.chdir(project_root)
dataset_name = 'bindingdb'
INPUT_CSV = f'./{dataset_name}/{dataset_name}_dataset.csv'  
OUTPUT_ESM_DIR = f'./{dataset_name}/esm2'
ESM_MODEL_NAME = 'esm2_t33_650M_UR50D'  
ESM_EMB_LAYER = 33

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(OUTPUT_ESM_DIR)

esm_model, esm_alphabet = pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
esm_model.eval()
esm_model = esm_model.cuda() if torch.cuda.is_available() else esm_model.cpu()

def preprocess_sequence(seq):
    return re.sub(r"[UZOB]", "X", seq.upper())

def extract_esm_features(sequence, unipoitID):
    seq_processed = preprocess_sequence(sequence)
    tokens = esm_alphabet.encode(seq_processed)
    token_len = len(tokens)

    max_esm_tokens = 1022
    if token_len > max_esm_tokens:
        tokens = tokens[:max_esm_tokens]

    token_tensor = torch.tensor([tokens]).to(next(esm_model.parameters()).device)
    with torch.no_grad():
        output = esm_model(
            token_tensor,
            repr_layers=[ESM_EMB_LAYER],
            return_contacts=False
        )

        esm_rep = output["representations"][ESM_EMB_LAYER][0, 0:token_len]

    esm_output_path = os.path.join(OUTPUT_ESM_DIR, f"{unipoitID}.pt")
    torch.save(esm_rep.cpu(), esm_output_path)
    print(f"[ESM] Saved: {esm_output_path}")

def main():
    with open(INPUT_CSV, mode='r', encoding='gbk') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in tqdm(reader, desc="Extracting features"):
            target_seq = row['Target'].strip()
            unipoit_id = row['target_key'].strip()
            extract_esm_features(target_seq, unipoit_id)

if __name__ == '__main__':
    main()