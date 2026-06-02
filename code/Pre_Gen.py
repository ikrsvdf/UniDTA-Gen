
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import os
import pickle
from utils import *
from tqdm import tqdm
from model import *
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
from rdkit import Chem
from utils import DataSet,collate_fn
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def format_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)
def test(model, test_loader, fold, config, input_df, tokenizer):
    model.eval()
    test_losses = []
    all_generated = []
    model.tokenizer = tokenizer
    
    criterion = nn.MSELoss()
    test_pbar = tqdm(test_loader, desc=f'Fold {fold}', leave=False)
    
    with torch.no_grad():
        for batch in test_pbar:
            try:
                smiles_seq = batch['smiles_seq'].to(device)
                affinity = batch['affinity'].to(device)
                esm = batch['esm_feat'].to(device)
                di = batch['3di_feat'].to(device)
                qed = batch['qed'].to(device)
                sas = batch['sas'].to(device)
                logp = batch['logp'].to(device)
                batched_graph = batch['graph']
                mask = create_mask_from_tensor(esm)
                esm = esm.float()
                di = di.float()             
                with torch.cuda.amp.autocast(enabled=False):
                    outputs, new_drug, lm_loss, kl_loss = model(
                        smiles_seq, esm, di, mask, 
                        affinity, qed, sas, logp, batched_graph.to(device)  
                    )
                    mse_loss = criterion(outputs, affinity)
                    loss = mse_loss + kl_loss * 0.001 + lm_loss
                generated_tokens, retry_count = model.generate(
                    smiles_seq, esm, di, mask, affinity, 
                    qed, sas, logp, 
                    random_sample=True,
                    max_retry=config.get('max_retry'),  
                    require_valid=config.get('require_valid', True),
                    graphs=batched_graph.to(device)
                )
                generated = tokenizer.get_text(generated_tokens)
                if config.get('filter', False):
                    generated = [format_smiles(smi) for smi in generated]
                    generated = [smi for smi in generated if smi]
                    print(generated)
                
                generated_str = ';'.join(generated) if generated else ""
                all_generated.append(generated_str)

                valid_count = len([smi for smi in generated if smi])
                test_pbar.set_postfix({
                    'test_loss': f'{loss.item():.4f}',
                    'retry': retry_count,
                    'valid': f'{valid_count}/{len(generated)}'
                })
                
                test_losses.append(loss.item())
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_generated.append("")
                continue
    
    
    print("Saving results...")
    output_path = os.path.join(f'/media/ubuntu/disk_1/lcg/DTA/code_原/results/parasite/P25779_generated.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_df['generated_molecules'] = all_generated
    if os.path.exists(output_path):
        os.remove(output_path)
    input_df.to_csv(output_path, index=False)
    
    return

def main():
    dataset = "parasite"
    project_root = '/media/ubuntu/disk_1/lcg/DTA'
    os.chdir(project_root)
    fold = "fold_0"
    grapg_dir = f'./{dataset}/saved_graphs'
    config = {
        'input_path': f'/media/ubuntu/disk_1/lcg/DTA/code_原/results/parasite/P25779_gen.csv',
        'filter': True,  
        'require_valid': True, 
        'max_retry': 10,  
    }

    input_df = pd.read_csv(config['input_path'])
    esm_dir = f'./{dataset}/esm2'
    di_dir = f'./{dataset}/3di_embeddings'
    
    with open(f'./{dataset}/{dataset}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    test_dataset = DataSet(
        f'/media/ubuntu/disk_1/lcg/DTA/code_原/results/parasite/P25779_gen.csv',
        esm_dir,
        di_dir,
        tokenizer=tokenizer,
        grapg_dir=grapg_dir,
        dataset =dataset
        
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,num_workers=0)
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
    
    pretrained_path = f"/media/ubuntu/disk_1/lcg/DTA/code/results/parasite/warm_start/fold_0/best_Gen_model_total.pth"
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict)

    test(model, test_loader, fold, config, input_df, tokenizer)

if __name__ == "__main__":

    main()