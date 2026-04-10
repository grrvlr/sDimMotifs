import numpy as np
import json 

def dict_to_mask(occurences_dict,signal_length):
    K=len(occurences_dict.keys())
    occurences_mask=np.zeros((K,signal_length))
    for i,key in enumerate(occurences_dict.keys()):
        occurences_list=occurences_dict[key]
        for occurence in occurences_list:
            s,e = occurence
            occurences_mask[i,s:e]+=1
    return occurences_mask

def mask_to_dict(L:np.ndarray)->list: 
    """Transfom binary mask to a list of start and ends

    Args:
        L (np.ndarray): binary mask, shape (n_label,length_time_series)

    Returns:
        list: start and end list. 
    """
    dict = {}
    for i,line in enumerate(L): 
        if np.count_nonzero(line)!=0:
            line = np.hstack(([0],line,[0]))
            diff = np.diff(line)
            start = np.where(diff==1)[0]+1
            end = np.where(diff==-1)[0]
            dict[str(i)] = [[int(s), int(e)] for s, e in zip(start, end)]
    return dict

def dict_to_dims(dims_dict):
    dims_list=[]
    for i,key in enumerate(dims_dict.keys()):
        dims_list.append(dims_dict[key])
    return dims_list

def multivariate_to_univariate_labels(labels, active_dimensions, n_d):
    """
    Transforme un tableau de labels multivariés en une liste de labels univariés (par dimension).

    Args:
        labels (np.ndarray): Tableau de forme (n_motifs, n_samples), contenant 0 ou 1 
                             selon la présence du motif à chaque instant (multivarié).
        active_dimensions (list[list[int]]): Liste de longueur n_motifs, contenant pour chaque motif
                                             les indices des dimensions actives.
        n_d (int): Nombre total de dimensions.

    Returns:
        list[np.ndarray]: Liste de longueur n_d. Chaque élément est un vecteur de taille (n_samples,)
                          contenant 0 par défaut, et le numéro du motif (index+1) si présent sur cette dimension.
    """

    n_motifs, n_samples = labels.shape
    # Initialisation : un label par dimension
    uni_labels = [[] for _ in range(n_d)]

    for motif_idx in range(n_motifs):
        motif_label = labels[motif_idx]
        dims = active_dimensions[motif_idx]
        for d in dims:
            uni_labels[d].append(motif_label)
            
    # Conversion en tableaux numpy et attribution des labels
    for d in range(n_d):
        uni_labels[d] = np.array(uni_labels[d], dtype=int)
    return uni_labels

def compute_configs_params(metadata_path):
    metadata = json.load(open(metadata_path,'r'))
    occ_pos=metadata['occurences_positions']
    active_dims=metadata['active_dims']
    all_lengths = []
    occ_per_pattern = {}
    for pid, occurrences in occ_pos.items():
        lengths = [end - start for start, end in occurrences]
        occ_per_pattern[pid] = len(lengths)
        all_lengths.extend(lengths)
    length_mean = float(np.mean(all_lengths)) if all_lengths else 0
    length_min  = int(np.min(all_lengths))    if all_lengths else 0
    length_max  = int(np.max(all_lengths))    if all_lengths else 0
    num_patterns = len(occ_pos)
    occ_counts = list(occ_per_pattern.values())
    max_occ_per_pattern = max(occ_counts) if occ_counts else 0
    dims_counts = [len(active_dims[pid]) for pid in active_dims]

    dims_min  = min(dims_counts) if dims_counts else 0
    dims_mean = float(np.mean(dims_counts)) if dims_counts else 0
    dims_max  = max(dims_counts) if dims_counts else 0
    return {
        "wlen_min": length_min,
        "wlen_mean": int(length_mean),
        "wlen_max": length_max,
        "n_patterns": num_patterns,
        "max_occ_per_pattern": max_occ_per_pattern,
        "dims_min": dims_min,
        "dims_mean": int(dims_mean),
        "dims_max": dims_max,
    }

import os

def compute_dataset_params(dataset_folder):
    """
    Agrège les stats sur TOUTES les occurrences de TOUT le dataset :
    - min global, max global
    - moyenne globale (et non la moyenne des moyennes)
    """

    metadata_files = [
        f for f in os.listdir(dataset_folder)
        if f.startswith("metadata") and f.endswith(".json")
    ]

    if not metadata_files:
        raise ValueError("❌ Aucun fichier metadata*.json trouvé")

    # Conteneurs globaux
    all_lengths   = []
    all_dims      = []
    all_n_patterns = []
    all_occ_counts = []

    for fname in metadata_files:
        metadata = json.load(open(os.path.join(dataset_folder, fname)))

        occ_pos = metadata["occurences_positions"]
        active_dims = metadata["active_dims"]

        # ----- longueurs -----
        for occurrences in occ_pos.values():
            for start, end in occurrences:
                all_lengths.append(end - start)

        # ----- nb de patterns -----
        all_n_patterns.append(len(occ_pos))

        # ----- occurrences par pattern -----
        for occurrences in occ_pos.values():
            all_occ_counts.append(len(occurrences))

        # ----- dimensions -----
        for dims in active_dims.values():
            all_dims.append(len(dims))

    # ----------------------------
    # Calculs globaux exacts
    # ----------------------------
    results = {
        "wlen_min":  int(np.min(all_lengths)),
        "wlen_mean": int(np.mean(all_lengths)),
        "wlen_max":  int(np.max(all_lengths)),

        "n_patterns_min":  int(np.min(all_n_patterns)),
        "n_patterns_mean": int(np.mean(all_n_patterns)),
        "n_patterns_max":  int(np.max(all_n_patterns)),
        
        "occ_per_pattern_max":  int(np.max(all_occ_counts)),

        "dims_min":  int(np.min(all_dims)),
        "dims_mean": int(np.mean(all_dims)),
        "dims_max":  int(np.max(all_dims)),
    }

    return results
