import pickle
import re
import pandas as pd
import torch
from tqdm import tqdm

class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>')
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])

    PATTEN = re.compile(r'\[[^\]]+\]'
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
        for a in tqdm(smiles_set, desc="Generating vocabs"):
            vocabs.update(re.findall(Tokenizer.PATTEN, a))
        return vocabs

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        #print(special_tokens)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda x: (len(x), x))
        #print(vocabs)
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles, return_atom_idx=False):
        l = []
        if return_atom_idx:
            atom_idx = []
        for i, s in enumerate(('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>')):
            if s not in self.s2i:
                a = 3  # <unk>
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


if __name__ == '__main__':
    df = pd.read_csv('/media/ubuntu/disk_1/lcg/DTA/bindingdb/bindingdb_dataset.csv')  
    smiles_list = df['Drug'].astype(str).tolist()
    vocabs = Tokenizer.gen_vocabs(smiles_list)
    print(vocabs)
    print(f"Vocab size (without special tokens): {len(vocabs)}")
    tokenizer = Tokenizer(vocabs)
    print(f"Total vocab size (with special tokens): {len(tokenizer)}")
    with open('/media/ubuntu/disk_1/lcg/DTA/bindingdb/bindingdb_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved")