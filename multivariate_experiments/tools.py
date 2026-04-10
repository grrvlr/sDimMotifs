import os 
import random
import pandas as pd
import json

random.seed(42)
initial_dir = os.getcwd()
os.chdir('/Users/valerio/Documents/Borelli/These/mPEPA')
from data.tools import create_armcoda_signal, create_armcoda_signal_from_idxes
os.chdir(initial_dir)


armcoda_path=os.getcwd()+'/armcoda'
if not os.path.exists(armcoda_path):
    os.mkdir(armcoda_path)

def create_scenario_dataset(scenario_name, mvts, active_dims: dict, interpolation='linear'):
    scenario_path=armcoda_path+'/'+scenario_name+'/Data'
    os.makedirs(scenario_path, exist_ok=True)
    for i in range(16):
        occurences_idxes_list=[]
        for j in range(len(mvts)):
            occurences_idxes_list.append([i,mvts[j],0])
            occurences_idxes_list.append([i,mvts[j],1])
        random.shuffle(occurences_idxes_list)
        signal, occurences_mask, occurences_positions=create_armcoda_signal_from_idxes(occurences_idxes_list,sparsity=0.5, sparsity_fluctuation=0.2,interpolation=interpolation)
        arm_signal=signal[:,54:]
        df_signal = pd.DataFrame(arm_signal)

        signal_path = os.path.join(scenario_path, f"signal_{i:02d}.csv")
        json_path = os.path.join(scenario_path, f"metadata_{i:02d}.json")

        df_signal.to_csv(signal_path, index=False)
        
        metadata = {
            "occurences_positions": occurences_positions,
            "active_dims": active_dims,
        }
        # Sauvegarde JSON
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)


import numpy as np
from itertools import combinations
from collections import defaultdict

def compute_intra_inter_motif_distances(X, y, occurences_idxes_list):
    """
    occurences_idxes_list = [ (motif_id, sample_idx), ... ]
    X has shape (n_files, d, n)
    """

    # ---- Extract occurrences ----
    motif_ids = np.array([int(float(occ[0])) for occ in occurences_idxes_list])
    occ_indices = np.array([int(occ[1]) for occ in occurences_idxes_list])

    occurences = X[occ_indices]  # shape (K, d, n)
    K, d, n = occurences.shape

    # ---- Normalize each occurrence per dim ----
    occ_norm = (occurences - occurences.mean(axis=2, keepdims=True)) / (
        occurences.std(axis=2, keepdims=True) + 1e-12
    )

    # ---- Distance matrix: (K,K,d) ----
    dist_matrix = np.zeros((K, K, d))

    for i, j in combinations(range(K), 2):
        for dim in range(d):
            seq_i = occ_norm[i, dim]
            seq_j = occ_norm[j, dim]
            dist = np.linalg.norm(seq_i - seq_j)

            dist_matrix[i, j, dim] = dist
            dist_matrix[j, i, dim] = dist

    # ----------------------------------------------------------------------
    #       COMPUTE INTRA & INTER MOTIF DISTANCES FOR EACH DIMENSION
    # ----------------------------------------------------------------------

    intra = defaultdict(list)   # motif_id → list of per-dim distances
    inter = defaultdict(list)   # motif_id → list of per-dim distances

    for i, j in combinations(range(K), 2):
        same_motif = (motif_ids[i] == motif_ids[j])
        m_id = motif_ids[i] if same_motif else None  # only meaningful for intra

        if same_motif:
            intra[m_id].append(dist_matrix[i, j])   # → vector of dim distances
        else:
            # inter distances: store per motif involved
            inter[motif_ids[i]].append(dist_matrix[i, j])
            inter[motif_ids[j]].append(dist_matrix[i, j])
    
    # ---- Convert lists of vectors → mean per dimension ----
    intra_mean = {}
    for m_id, distances in intra.items():
        intra_mean[m_id] = np.mean(np.stack(distances), axis=0)  # shape (d,)

    inter_mean = {}
    for m_id, distances in inter.items():
        inter_mean[m_id] = np.mean(np.stack(distances), axis=0)  # shape (d,)

    return dist_matrix, intra_mean, inter_mean
