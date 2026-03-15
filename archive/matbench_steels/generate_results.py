import os
import torch
import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Composition
import logging

# We will need the featurizers and model definitions from trm13/trm14
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Training Code"))

# We are going to submit the V13A model since it is the SOTA ensemble
from trm13 import DeepHybridTRM, ExpandedFeaturizer, SEEDS, strat_split, DSData, dl_kw

logging.basicConfig(level=logging.INFO, format='%(name)s │ %(message)s')
log = logging.getLogger("MatbenchSub")

def generate_submission():
    log.info("Starting Matbench benchmark recording for V13A...")
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_steels"])
    task = list(mb.tasks)[0]
    task.load()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare entire dataset features once (just like trm13.py does)
    comps_all = [Composition(c) for c in task.df.index]
    feat = ExpandedFeaturizer(cache="../mat2vec_cache")
    X_all_raw = feat.featurize_all(comps_all)
    n_extra = feat.n_extra

    # Configuration for V13A
    model_kw = dict(n_props=22, stat_dim=6, n_extra=n_extra,
                    mat2vec_dim=200, d_attn=64, nhead=4,
                    d_hidden=96, ff_dim=150, dropout=0.2, max_steps=20)
    cname = 'V13A-2xSA-StdDS'

    for fold_idx, (train_df, test_df) in enumerate(task.folds):
        log.info(f"Processing Fold {fold_idx + 1}/5")
        
        # We must re-create the standard scaling exactly as done in training
        # The Matbench API provides train_df and test_df as dataframes.
        # However, to be perfectly consistent with our saved models which 
        # used specific train/val splits to fit the scaler, we will match the 
        # exact indexing from trm13.py
        
        # In trm13.py, folds were generated via:
        # kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
        # We can just read the task's indices to identify tv_i and te_i
        
        tv_i = np.where(task.df.index.isin(train_df.index))[0]
        te_i = np.where(task.df.index.isin(test_df.index))[0]
        targets_all = task.df.values.astype(np.float32)

        predicted_ensemble = []
        for seed in SEEDS:
            # Recreate the exact split to fit the correct scaler
            tri, vli = strat_split(targets_all[tv_i], 0.15, seed + fold_idx)
            
            # The scaler was heavily mutated during the loop due to calling fit_scaler repeatedly
            # Let's cleanly instantiate a new instance for this specific fold/seed 
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_all_raw[tv_i][tri])
            
            te_s = np.nan_to_num(scaler.transform(X_all_raw[te_i]), nan=0.0).astype(np.float32)
            te_s_tensor = torch.tensor(te_s, dtype=torch.float32).to(device)

            model = DeepHybridTRM(**model_kw).to(device)
            model_path = f"../Training Code/trm_models_v13/{cname}_seed{seed}_fold{fold_idx+1}.pt"
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state['model_state'])
            model.eval()

            with torch.no_grad():
                pred = model(te_s_tensor).cpu().numpy()
            predicted_ensemble.append(pred)

        # Average the predictions
        fold_predictions = np.mean(predicted_ensemble, axis=0)
        
        # Record with Matbench
        task.record(fold_idx, fold_predictions)

    # Save benchmark
    log.info("Saving results.json.gz...")
    mb.to_file("results.json.gz")
    log.info("Done!")

if __name__ == "__main__":
    generate_submission()
