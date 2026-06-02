import os
import pandas as pd
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, set_seed
import torch
import re
from tqdm import tqdm
import time

project_root = '/media/ubuntu/disk_1/lcg/DTA'
os.chdir(project_root)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
set_seed(0)
dataname = 'bindingdb'
tokenizer = T5Tokenizer.from_pretrained('/media/ubuntu/disk_1/lcg/LLM/Prostt5', do_lower_case=False)
model = AutoModelForSeq2SeqLM.from_pretrained('/media/ubuntu/disk_1/lcg/LLM/Prostt5').to(device)
model.float() if device.type == 'cpu' else model.half()

def generate_3di_sequence(aa_sequence):
    if not aa_sequence or not isinstance(aa_sequence, str):
        return ""
    
    sequence_example = [aa_sequence]
    min_len = min([len(s) for s in sequence_example]) + 1
    max_len = max([len(s) for s in sequence_example]) + 1
    
    sequence_example = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_example]
    sequence_example = ["<AA2fold>" + " " + s for s in sequence_example]
    
    ids = tokenizer.batch_encode_plus(sequence_example,
                                      add_special_tokens=True,
                                      padding="longest",
                                      return_tensors='pt').to(device)
    
    gen_kwargs_aa2fold = {
        "do_sample": True,
        "num_beams": 3,
        "top_p": 0.95,
        "temperature": 1.2,
        "top_k": 6,
        "repetition_penalty": 1.2,
    }
    
    with torch.no_grad():
        translations = model.generate(
            ids.input_ids,
            attention_mask=ids.attention_mask,
            max_length=max_len,
            min_length=min_len,
            early_stopping=True,
            num_return_sequences=1,
            **gen_kwargs_aa2fold
        )
    
    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    structure_sequence = "".join(decoded_translations[0].split(" "))
    return structure_sequence

df = pd.read_csv(f'./{dataname}/{dataname}_dataset.csv')
print(f"{len(df)}")

original_count = len(df)
df_unique = df.drop_duplicates(subset=['Target']).copy()
unique_count = len(df_unique)

cache = {}

def generate_with_cache(aa_sequence):
    if aa_sequence in cache:
        return cache[aa_sequence]
    
    result = generate_3di_sequence(aa_sequence)
    cache[aa_sequence] = result
    return result

start_time = time.time()
targets = df_unique['Target'].tolist()
structure_sequences = []

for target in tqdm(targets, desc="Generating 3di sequences", unit="seq"):
    structure_seq = generate_with_cache(target)
    structure_sequences.append(structure_seq)

end_time = time.time()
processing_time = end_time - start_time
df_unique['3di_sequence'] = structure_sequences
target_to_3di = dict(zip(df_unique['Target'], df_unique['3di_sequence']))
df['3di_sequence'] = df['Target'].map(target_to_3di)
full_output_path = f'./{dataname}/{dataname}_dataset_with_3di.csv'
df.to_csv(full_output_path, index=False)
print("\n" + "="*50)
print("Processing completed! Statistics:")
print(f"Original data count: {original_count}")
print(f"Unique sequence count: {unique_count}")
print(f"Duplicate sequence count: {original_count - unique_count}")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Average time per sequence: {processing_time/unique_count:.2f} seconds")
print(f"Complete data (including duplicates) saved to: {full_output_path}")
print("="*50)




