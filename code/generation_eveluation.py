import pandas as pd
import argparse
from rdkit import Chem
import os
from datetime import datetime

project_root = '/media/ubuntu/disk_1/lcg/DTA'
os.chdir(project_root)

def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is chemically valid."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False


def evaluate_smiles(smiles_list, reference_set=None):
    """Evaluate validity, uniqueness, and novelty of SMILES."""
    valid = [s for s in smiles_list if is_valid_smiles(s)]
    s_valid_smiles = set(valid)
    
    validity_ratio = len(valid) / len(smiles_list) if smiles_list else 0
    

    uniqueness_ratio = len(s_valid_smiles) / len(valid) if valid else 0

    if reference_set is not None:
        novelty_ratio = len(s_valid_smiles - reference_set) / len(s_valid_smiles) if s_valid_smiles else 0
        novel = [s for s in s_valid_smiles if s not in reference_set]
    else:
        novelty_ratio = 0
        novel = list(s_valid_smiles)
    
    ratio_available_molecules = len(novel) / len(smiles_list) if smiles_list else 0

    return {
        "validity_ratio": validity_ratio,
        "uniqueness_ratio": uniqueness_ratio,
        "novelty_ratio": novelty_ratio,
        "ratio_of_available_molecules": ratio_available_molecules,
        "total_generated": len(smiles_list),
        "valid_count": len(valid),
        "unique_count": len(s_valid_smiles),
        "novel_count": len(novel)
    }


def save_results_to_csv(results, output_path, dataset_name, fold_number=0):
    """Save evaluation results to a CSV file."""
    result_dict = {
        'dataset': [dataset_name],
        'fold': [fold_number],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'validity_ratio': [results['validity_ratio']],
        'uniqueness_ratio': [results['uniqueness_ratio']],
        'novelty_ratio': [results['novelty_ratio']],
        'ratio_of_available_molecules': [results['ratio_of_available_molecules']],
        'total_generated': [results['total_generated']],
        'valid_count': [results['valid_count']],
        'unique_count': [results['unique_count']],
        'novel_count': [results['novel_count']]
    }
    
    df_results = pd.DataFrame(result_dict)
    
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_results], ignore_index=True)
        df_combined.to_csv(output_path, index=False)
        print(f"Results appended to existing file: {output_path}")
    else:
        df_results.to_csv(output_path, index=False)
        print(f"Results saved to new file: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bindingdb', help='Dataset name (default: "bindingdb")')
    parser.add_argument('--fold', type=int, default=4, help='Fold number (default: 0)')

    args = parser.parse_args()
    # output_dir = f"/DTA/code/results/{args.dataset}/fold_{args.fold}"
    # os.makedirs(output_dir, exist_ok=True)
    
    file_path = f"/media/ubuntu/disk_1/lcg/DTA/data_process/generated.csv"
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    df = pd.read_csv(file_path)
    if 'generated_molecules' not in df.columns:
        print("Error: Column 'generated_molecules' not found in the dataset.")
        return

    generated = df['generated_molecules'].tolist()
    print(f"Total generated molecules: {len(generated)}")
    
    if 'Drug' in df.columns:
        reference_set = set(df['Drug'])
        print(f"Reference set size: {len(reference_set)}")
    else:
        print("Warning: 'Drug' column not found. Novelty will be calculated without reference set.")
        reference_set = None


    results = evaluate_smiles(generated, reference_set)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset}, Fold: {args.fold}")
    print(f"Total generated: {results['total_generated']}")
    print(f"Valid molecules: {results['valid_count']}")
    print(f"Unique molecules: {results['unique_count']}")
    print(f"Novel molecules: {results['novel_count']}")
    print("-"*50)
    print(f"Validity Ratio: {results['validity_ratio']:.6f}")
    print(f"Uniqueness Ratio: {results['uniqueness_ratio']:.6f}")
    print(f"Novelty Ratio: {results['novelty_ratio']:.6f}")
    print(f"Available Molecules Ratio: {results['ratio_of_available_molecules']:.6f}")
    print("="*50)

    # output_filename = f"evaluation_results.csv"
    # output_path = os.path.join(output_dir, output_filename)
    
    #save_results_to_csv(results, output_path, args.dataset, args.fold)
    #print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
