import re
import torch
import pandas as pd
import os
from transformers import T5Tokenizer, T5EncoderModel
from collections import defaultdict
import tqdm

project_root = '/media/ubuntu/disk_1/lcg/DTA'
os.chdir(project_root)
dataset_name = 'bindingdb'
save_dir = f'./{dataset_name}/3di_embeddings'
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('/media/ubuntu/disk_1/lcg/LLM/Prostt5', do_lower_case=False)
model = T5EncoderModel.from_pretrained('/media/ubuntu/disk_1/lcg/LLM/Prostt5').to(device)
model = model.half() if device != 'cpu' else model.float()
model.eval()
csv_file_path = f'./{dataset_name}/{dataset_name}_dataset_with_3di.csv'
df = pd.read_csv(csv_file_path)
if not all(col in df.columns for col in ['3di_sequence', 'target_key']):
    raise ValueError("CSV file must contain '3di_sequence' and 'target_key' columns")

id_groups = defaultdict(list)
for _, row in df.iterrows():
    id_groups[str(row['target_key'])].append(row['3di_sequence'])

duplicate_ids = {id_: seqs for id_, seqs in id_groups.items() if len(seqs) > 1}

processed_ids = set()
total_to_process = len(df)

pbar = tqdm.tqdm(total=total_to_process, desc="Generating 3di embeddings")

for index, row in df.iterrows():
    target_seq = row['3di_sequence']
    uniprot_id = str(row['target_key'])
    if uniprot_id in processed_ids:
        pbar.update(1)
        continue
    processed_ids.add(uniprot_id)
    cleaned_seq = " ".join(list(re.sub(r"[UZOB]", "X", target_seq)))
    prefix = "<AA2fold> " if cleaned_seq.replace(" ", "").isupper() else "<fold2AA> "
    input_text = prefix + cleaned_seq
    try:
        inputs = tokenizer(
            [input_text],
            add_special_tokens=True,
            padding="longest",
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = embedding_rpr.last_hidden_state
        emb_0 = last_hidden_state[0, 1:-1]  
        output_filename = f"{uniprot_id}.pt"
        output_path = os.path.join(save_dir, output_filename)
        torch.save(emb_0.cpu(), output_path)
        pbar.set_postfix({"Latest saved": f"{uniprot_id}", "Shape": str(emb_0.shape)})
    except Exception as e:
        print(f"Error processing {uniprot_id}: {e}")
    pbar.update(1)
pbar.close()
print(f"\nProcessing completed!")
print(f"Total unique IDs processed: {len(processed_ids)}")
print(f"Embedding files saved in: {save_dir}")


