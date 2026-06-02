import pickle
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from utils import DataSet, collate_fn,calculate_metrics,Tokenizer
from Radam import RAdam
from dgl.dataloading import GraphDataLoader
from lookahead import Lookahead
from tqdm import tqdm
from sklearn.model_selection import KFold
from model import KA_GAT, DrugGenModel,create_mask_from_tensor
import warnings
from sklearn.metrics import mean_squared_error
import argparse
import multiprocessing as mp
warnings.filterwarnings("ignore", category=UserWarning)
project_root = '/media/ubuntu/disk_1/lcg/DTA'
os.chdir(project_root)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)



def train_Gen_DTA(model, train_loader, test_loader, num_epochs=300, learning_rate=0.001, dataset=None,fold=None,start_set=None):
    criterion = nn.MSELoss()
    weight_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(param)  
        else:
            weight_params.append(param)  

    inner_optimizer = RAdam(
        [{'params': weight_params, 'weight_decay': 1e-4},
         {'params': bias_params, 'weight_decay': 0.0}],
        lr=learning_rate,
        betas=(0.9, 0.999)
    )
    optimizer = Lookahead(
        inner_optimizer,
        la_steps=5,  
        la_alpha=0.5  
    )
    
    results_dir = f'./code/results/{dataset}/{start_set}/{fold}'
    os.makedirs(results_dir, exist_ok=True)
    eval_file = os.path.join(results_dir, 'results.txt')
    if not os.path.exists(eval_file):
        open(eval_file, 'w').close()
    csv_path = os.path.join(results_dir, 'best_test_predictions.csv')
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=5e-4, last_epoch=-1)
    
    # 添加混合精度训练支持
    scaler = torch.cuda.amp.GradScaler()
    
    best_test_mse_loss = float('inf')
    best_test_total_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_mse_losses = []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Training]', leave=False)
        
        for batch in train_pbar:
            smiles_seq = batch['smiles_seq'].to(device)
            affinity = batch['affinity'].to(device)
            qed = batch['qed'].to(device)
            sas = batch['sas'].to(device)
            logp = batch['logp'].to(device)
            batched_graph = batch['graph'].to(device) 
            
            esm = batch['esm_feat'].to(device)
            di = batch['3di_feat'].to(device)
            mask = create_mask_from_tensor(esm)
            esm = esm.float()
            di = di.float()
            
            optimizer.zero_grad()
            
            # 使用混合精度训练加速
            with torch.cuda.amp.autocast():
                outputs, new_drug, lm_loss, kl_loss = model(
                    smiles_seq, esm, di, mask, 
                    affinity, qed, sas, logp, batched_graph
                )
                mse_loss = criterion(outputs, affinity)
                loss = mse_loss + kl_loss * 0.001 + lm_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            train_mse_losses.append(mse_loss.item())
            train_pbar.set_postfix({
                'train_loss': f'{loss.item():.4f}',
                'train_mse': f'{mse_loss.item():.4f}',
                'kl_loss': f'{kl_loss.item():.4f}',
                'lm_loss': f'{lm_loss.item():.4f}'
            })
        
        schedule.step()
        
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
        
        model.eval()
        test_losses = []
        test_mse_losses = []
        test_true = []
        test_pred = []
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Testing]', leave=False)
        
        with torch.no_grad():
            for batch in test_pbar:
                smiles_seq = batch['smiles_seq'].to(device)
                affinity = batch['affinity'].to(device)
                qed = batch['qed'].to(device)
                sas = batch['sas'].to(device)
                logp = batch['logp'].to(device)
                batched_graph = batch['graph'].to(device)
                
                esm = batch['esm_feat'].to(device)
                di = batch['3di_feat'].to(device)
                mask = create_mask_from_tensor(esm)
                esm = esm.float()
                di = di.float()
                
                with torch.cuda.amp.autocast(enabled=False):
                    outputs, new_drug, lm_loss, kl_loss = model(
                        smiles_seq, esm, di, mask, 
                        affinity, qed, sas, logp, batched_graph  # 传入批量图
                    )
                    mse_loss = criterion(outputs, affinity)
                    loss = mse_loss + kl_loss * 0.001 + lm_loss
                
                test_losses.append(loss.item())
                test_mse_losses.append(mse_loss.item())
                test_true.append(affinity.cpu().numpy().flatten())
                test_pred.append(outputs.cpu().numpy().flatten())
                
                test_pbar.set_postfix({
                    'test_loss': f'{loss.item():.4f}',
                    'test_mse': f'{mse_loss.item():.4f}',
                    'kl_loss': f'{kl_loss.item():.4f}',
                    'lm_loss': f'{lm_loss.item():.4f}'
                })
        
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        metrics = calculate_metrics(test_true, test_pred)
        test_mse = mean_squared_error(test_true, test_pred)
        avg_train_mse = np.mean(train_mse_losses)
        
        tqdm.write(f'Test Metrics:')
        for metric, value in metrics.items():
            tqdm.write(f'{metric}: {value:.4f}')
        
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], '
                   f'Train Loss: {avg_train_loss:.4f}, Train MSE: {avg_train_mse:.4f}, '
                   f'Test Loss: {avg_test_loss:.4f}, Test MSE: {test_mse:.4f}')

        if test_mse < best_test_mse_loss:
            best_test_mse_loss = test_mse
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_Gen_model_mse.pth'))
            tqdm.write(f'  → New best MSE model saved! (MSE: {best_test_mse_loss:.4f})')
            best_true = test_true
            best_pred = test_pred
            best_metrics = metrics
            
            with open(eval_file, 'a') as f:
                f.write(f"\nBest evaluation at epoch {epoch + 1}:\n")
                for metric, value in best_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")

            results_df = pd.DataFrame({
                'True_Value': best_true,
                'Predicted_Value': best_pred
            })
            results_df.to_csv(csv_path, index=False)
            tqdm.write(f'Best predictions saved to {csv_path}')
        

        if avg_test_loss < best_test_total_loss:
            best_test_total_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_Gen_model_total.pth'))
            tqdm.write(f'  → New best Total Loss model saved! (Total Loss: {best_test_total_loss:.4f})')
    

    return best_test_mse_loss, best_test_total_loss


def main():
    parser = argparse.ArgumentParser(description='Drug-Target Affinity Prediction')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--fold', type=str, default='fold_3')
    parser.add_argument('--start_set', type=str, default='protein_coldstart')
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    dataset = args.dataset
    fold = args.fold
    start_set = args.start_set
    esm_dir = f'./{dataset}/esm2'
    di_dir = f'./{dataset}/3di_embeddings'
    grapg_dir = f'./{dataset}/saved_graphs'
    with open(f'./{dataset}/{dataset}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"{'Batch Size':<20}: {batch_size}")
    print(f"{'Num Epochs':<20}: {num_epochs}")
    print(f"{'Learning Rate':<20}: {learning_rate}")
    print(f"{'Dataset':<20}: {dataset}")
    print(f"{'Fold':<20}: {fold}")
    print(f"{'tokenizer':<20}: {len(tokenizer)}")



    train_dataset = DataSet(
        f'./{dataset}/data_folds/{start_set}/{fold}/train.csv',
        esm_dir,
        di_dir,
        tokenizer=tokenizer, 
        grapg_dir=grapg_dir,
        dataset=dataset
    )

    test_dataset = DataSet(
        f'./{dataset}/data_folds/{start_set}/{fold}/test.csv',
        esm_dir,
        di_dir,
        tokenizer=tokenizer, 
        grapg_dir=grapg_dir,
        dataset=dataset
    )
    
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

    g = torch.Generator()
    g.manual_seed(42)
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, generator=g)
    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, generator=g)
    
    model = DrugGenModel(
        tokenizer=tokenizer,
        ka_gat_model=ka_gat_model
    ).to(device)
    train_Gen_DTA(
        model, train_loader, test_loader, num_epochs, learning_rate, dataset=dataset,fold=fold,start_set=start_set
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  
    main()
