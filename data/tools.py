import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import json 
import random
import json

from scipy.interpolate import CubicSpline

arm_coda_path= os.getcwd()+ '/data/arm-CODA-raw-dataset/dataset/'

def spline_random_control(start, end, L, n_ctrl=2, ctrl_scale=0.2):
    n_d = start.shape[0]

    xs = np.concatenate(([0], np.sort(np.random.rand(n_ctrl)), [1]))

    controls = np.zeros((n_ctrl+2, n_d))
    controls[0] = start
    controls[-1] = end

    for i, a in enumerate(xs[1:-1], start=1):
        controls[i] = start + a*(end-start) + np.random.normal(scale=ctrl_scale, size=n_d)

    t = np.linspace(0, 1, L)
    result = np.zeros((L, n_d))

    for d in range(n_d):
        cs = CubicSpline(xs, controls[:, d])
        result[:, d] = cs(t)

    return result

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

def extract_movement_occurences(subject_id,movement_id,path=arm_coda_path):
    file_name=path+'armcoda_subject'+str(subject_id)+'_movement' +str(movement_id)
    data=pd.read_csv(file_name+'.csv').to_numpy()
    with open(file_name+'.json','r') as f:
        metadata=json.load(f)

    s1,e1=metadata["Movement_label"]["Iteration_1"]
    s2,e2=metadata["Movement_label"]["Iteration_2"]  
    occ_lengths=[e1-s1,e2-s2] 

    return ([data[s1:e1]-data[s1],data[s2:e2]-data[s2]], occ_lengths)

def create_armcoda_signal(n_motifs, subject_ids=['0'], snr_motifs_db=100,snr_gaps_db=50,
                                sparsity=0.1, sparsity_fluctuation=0.2,
                                save=False, id=0, path=arm_coda_path):
    """
    Crée un signal synthétique à partir de motifs extraits de plusieurs sujets.

    Args:
        n_motifs (int): Nombre de motifs distincts à inclure
        subject_ids (list): Liste des IDs des sujets à inclure
        
        
        sparsity (float): Densité moyenne des motifs
        sparsity_fluctuation (float): Variabilité de la densité
        n_d (int): Dimensions du signal (par ex: 3 * nb marqueurs)
        save (bool): Sauvegarder les fichiers de sortie
        id (int): Identifiant de sauvegarde

    Returns:
        signal (ndarray): Signal synthétique généré
        occurences_mask (ndarray): Masque d'occurrences
        occurences_idxes_list (list): Liste des occurrences utilisées (triplets)
    """
    occurences_list = []
    occurences_idxes_list = []
    max_occ_length = 0

    # Choisir aléatoirement les motifs (mvt_idx) parmi 15 possibles
    all_mvt_idx = np.arange(15)
    random.shuffle(all_mvt_idx)
    mvts_idxes = all_mvt_idx[:n_motifs]

    # Extraire les occurrences pour chaque sujet et chaque motif
    for mvt_idx in mvts_idxes:
        for subj_id in subject_ids:
            occ, occ_lengths = extract_movement_occurences(subj_id, mvt_idx,path)
            occurences_list += [occ[0], occ[1]]  # Prendre les deux premières occurrences
            occurences_idxes_list += [[subj_id, mvt_idx, 1], [subj_id, mvt_idx, 2]]
            max_occ_length = max(max_occ_length, occ_lengths[0], occ_lengths[1])

    n_d=occurences_list[0].shape[1]
    # Mélanger
    shuffle_indices = np.arange(len(occurences_list))
    random.shuffle(shuffle_indices)
    occurences_list = [occurences_list[i] for i in shuffle_indices]
    occurences_idxes_list = np.array(occurences_idxes_list)[shuffle_indices]

    all_motifs_concat = np.concatenate(occurences_list, axis=0)
    power_signal = np.mean(all_motifs_concat ** 2)

    noise_power_motifs = power_signal / (10 ** (snr_motifs_db / 10))
    noise_power_gaps = power_signal / (10 ** (snr_gaps_db / 10))

    sigma_motifs = np.sqrt(noise_power_motifs)
    sigma_gaps = np.sqrt(noise_power_gaps)

    # Construction du signal
    initial_gap = int(max_occ_length * sparsity + np.random.uniform(-1, 1) * sparsity_fluctuation)
    signal = np.random.normal(scale=sigma_gaps, size=(initial_gap, n_d))

    occurences_positions = {}
    for i, occ in enumerate(occurences_list):
        noisy_occ = occ + np.random.normal(scale=sigma_motifs, size=occ.shape)
        s1 = signal.shape[0]
        e1 = s1 + occ.shape[0]
        signal = np.concatenate((signal, noisy_occ), axis=0)

        label = occurences_idxes_list[i][1]
        occurences_positions.setdefault(label, []).append([s1, e1])

        gap_length = int(max_occ_length * sparsity + np.random.uniform(-1, 1) * sparsity_fluctuation)
        last_value = noisy_occ[-1]  # shape (n_d,)
        # interpolation linéaire entre last_value et 0 sur gap_length points
        linear = np.linspace(last_value, np.zeros_like(last_value), gap_length)
        gap = linear + np.random.normal(scale=sigma_gaps, size=(gap_length, n_d))
        signal = np.concatenate((signal, gap), axis=0)

    occurences_mask = dict_to_mask(occurences_positions, signal.shape[0])
    
    # Sauvegarde optionnelle
    if save:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'arm_coda_synthetic')
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(signal).to_csv(os.path.join(output_dir, f'signal_{id}.csv'), index=False)
        pd.DataFrame(occurences_mask).to_csv(os.path.join(output_dir, f'occurences_mask_{id}.csv'), index=False)
        with open(os.path.join(output_dir, f'occurences_idxes_{id}.json'), "w") as f:
            json.dump({"occurences_idxes": occurences_idxes_list}, f)
    return signal, occurences_mask, occurences_idxes_list.tolist()

def create_armcoda_signal_from_idxes(occurences_idxes_list, snr_motifs_db=100,snr_gaps_db=50,
                             sparsity=1, sparsity_fluctuation=0.2,
                             save=False, interpolation= 'linear', id=0,path=arm_coda_path):
    """
    Reconstruit un signal synthétique à partir d'une liste d'index d'occurrences
    incluant l'information du sujet.

    Args:
        occurences_idxes_list (list): liste de [subject_id, mvt_idx, occ_id] pour chaque occurrence
        noise_amplitude (float): Amplitude du bruit ajouté
        sparsity (float): Espacement moyen entre les occurrences
        sparsity_fluctuation (float): Variabilité de l'espacement
        n_d (int): Nombre de dimensions du signal (ex: 3 * n_markers)
        save (bool): Si True, enregistre les fichiers
        id (int): Identifiant pour le nom de fichier si sauvegarde

    Returns:
        signal (np.ndarray): Le signal reconstruit
        occurences_mask (np.ndarray): Masque des occurrences
        occurences_idxes_list (list): La liste utilisée
    """
    occurences_list = []
    max_occ_length = 0

    # Charger toutes les occurrences à partir des triplets [subject_id, mvt_idx, occ_id]
    for subject_id, mvt_idx, occ_id in occurences_idxes_list:
        occ, occ_lengths = extract_movement_occurences(subject_id, mvt_idx,path)
        occurence = occ[occ_id - 1]  # occ_id est supposé commencer à 1
        occurences_list.append(occurence)
        max_occ_length = max(max_occ_length, len(occurence))

    n_d=occurences_list[0].shape[1]
    all_motifs_concat = np.concatenate(occurences_list, axis=0)
    power_signal = np.mean(all_motifs_concat ** 2)

    noise_power_motifs = power_signal / (10 ** (snr_motifs_db / 10))
    noise_power_gaps = power_signal / (10 ** (snr_gaps_db / 10))

    sigma_motifs = np.sqrt(noise_power_motifs)
    sigma_gaps = np.sqrt(noise_power_gaps)

    # Construction du signal
    initial_gap = int(max_occ_length * sparsity + np.random.uniform(-1, 1) * sparsity_fluctuation)
    signal = np.random.normal(scale=sigma_gaps, size=(initial_gap, n_d))

    occurences_positions = {}
    for i, occ in enumerate(occurences_list):
        noisy_occ = occ + np.random.normal(scale=sigma_motifs, size=occ.shape)
        s1 = signal.shape[0]
        e1 = s1 + occ.shape[0]
        signal = np.concatenate((signal, noisy_occ), axis=0)

        label = occurences_idxes_list[i][1]
        occurences_positions.setdefault(label, []).append([s1, e1])

        gap_length = int(max_occ_length * sparsity + np.random.uniform(-1, 1) * sparsity_fluctuation)
        last_value = noisy_occ[-1]  # shape (n_d,)
        if interpolation == 'random':
            gap = np.random.normal(scale=sigma_gaps, size=(gap_length, n_d))
        elif interpolation == 'linear':
        # interpolation linéaire entre last_value et 0 sur gap_length points
            linear = np.linspace(last_value, np.zeros_like(last_value), gap_length)
            gap = linear + np.random.normal(scale=sigma_gaps, size=(gap_length, n_d))
        elif interpolation == 'spline':
            spline = spline_random_control(last_value, np.zeros_like(last_value), gap_length)
            gap = spline + np.random.normal(scale=sigma_gaps, size=(gap_length, n_d))
        signal = np.concatenate((signal, gap), axis=0)

    occurences_mask = dict_to_mask(occurences_positions, signal.shape[0])

    if save:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'arm_coda_synthetic')
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(signal).to_csv(os.path.join(output_dir, f'signal_{id}.csv'), index=False)
        pd.DataFrame(occurences_mask).to_csv(os.path.join(output_dir, f'occurences_mask_{id}.csv'), index=False)
        with open(os.path.join(output_dir, f'occurences_idxes_{id}.json'), "w") as f:
            json.dump({"occurences_idxes": occurences_idxes_list}, f)

    return signal, occurences_mask, occurences_positions 

def create_armcoda_signal_from_idxes_dims_positions(occurences_idxes_list, active_dims_list, positions=None, sigma_gaps=1, sigma_motifs=1,
                             save=False, id=0,path=arm_coda_path):
    """
    Reconstruit un signal synthétique à partir d'une liste d'index d'occurrences
    incluant l'information du sujet.

    Args:
        occurences_idxes_list (list): liste de [subject_id, mvt_idx, occ_id] pour chaque occurrence
        noise_amplitude (float): Amplitude du bruit ajouté
        sparsity (float): Espacement moyen entre les occurrences
        sparsity_fluctuation (float): Variabilité de l'espacement
        n_d (int): Nombre de dimensions du signal (ex: 3 * n_markers)
        save (bool): Si True, enregistre les fichiers
        id (int): Identifiant pour le nom de fichier si sauvegarde

    Returns:
        signal (np.ndarray): Le signal reconstruit
        occurences_mask (np.ndarray): Masque des occurrences
        occurences_idxes_list (list): La liste utilisée
    """
    occurences_list = []
    max_occ_length = 0

    # Charger toutes les occurrences à partir des triplets [subject_id, mvt_idx, occ_id]
    for subject_id, mvt_idx, occ_id in occurences_idxes_list:
        occ, occ_lengths = extract_movement_occurences(subject_id, mvt_idx,path)
        occurence = occ[occ_id - 1]  # occ_id est supposé commencer à 1
        occurences_list.append(occurence)
        max_occ_length = max(max_occ_length, len(occurence))

    n_d=occurences_list[0].shape[1]
    # longueur totale du signal
    if positions is None:
        raise ValueError("positions must be provided to allow overlapping motifs")

    signal_length = max(
        pos + occ.shape[0]
        for pos, occ in zip(positions, occurences_list)
    )

    # signal bruité global
    signal = np.random.normal(scale=sigma_gaps, size=(signal_length, n_d))

    # masque subdimensionnel
    occurences_mask = np.zeros((signal_length, n_d), dtype=int)

    # injection des motifs
    occurences_positions = {}
    for i, occ in enumerate(occurences_list):
        dims = active_dims_list[i]
        pos = positions[i]

        noisy_occ = occ[:, dims] + np.random.normal(
            scale=sigma_motifs,
            size=(occ.shape[0], len(dims))
        )
        signal[pos:pos+occ.shape[0], dims] += noisy_occ
        occurences_mask[pos:pos+occ.shape[0], dims] = i + 1

        label = occurences_idxes_list[i][1]
        s1= pos
        e1= pos + occ.shape[0]
        occurences_positions.setdefault(label, []).append([s1, e1])

    occurences_mask = dict_to_mask(occurences_positions, signal.shape[0])

    if save:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'arm_coda_synthetic')
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(signal).to_csv(os.path.join(output_dir, f'signal_{id}.csv'), index=False)
        pd.DataFrame(occurences_mask).to_csv(os.path.join(output_dir, f'occurences_mask_{id}.csv'), index=False)
        with open(os.path.join(output_dir, f'occurences_idxes_{id}.json'), "w") as f:
            json.dump({"occurences_idxes": occurences_idxes_list}, f)

    return signal, occurences_mask, occurences_positions 

def create_aeon_signal(X, y, n_motifs, n_occurences, snr_motifs_db=100,snr_gaps_db=50,
                                sparsity=0.1, sparsity_fluctuation=0.2,
                                id=0, interpolation= 'linear'):
    """
    Crée un signal synthétique à partir de motifs extraits de plusieurs sujets.

    Args:
        n_motifs (int): Nombre de motifs distincts à inclure
        subject_ids (list): Liste des IDs des sujets à inclure
        
        
        sparsity (float): Densité moyenne des motifs
        sparsity_fluctuation (float): Variabilité de la densité
        n_d (int): Dimensions du signal (par ex: 3 * nb marqueurs)
        save (bool): Sauvegarder les fichiers de sortie
        id (int): Identifiant de sauvegarde

    Returns:
        signal (ndarray): Signal synthétique généré
        occurences_mask (ndarray): Masque d'occurrences
        occurences_idxes_list (list): Liste des occurrences utilisées (triplets)
    """
    occurences_list = []
    occurences_idxes_list = []
    max_occ_length = 0

    # Choisir aléatoirement les motifs (mvt_idx) parmi les motifs possibles
    all_mvt_idx = np.unique(y)
    random.shuffle(all_mvt_idx)
    mvts_idxes = all_mvt_idx[:n_motifs]

    # Extraire les occurrences pour chaque sujet et chaque motif
    for mvt_idx in mvts_idxes:
        occurences_idxes=np.where(y==mvt_idx)[0]
        random.shuffle(occurences_idxes)
        occurences_idxes=occurences_idxes[:n_occurences]
        for pos in occurences_idxes:
            occ=X[pos,:,:].T
            occ_length=occ.shape[0] 
            occurences_list.append(occ-occ[0,:])
            occurences_idxes_list.append([mvt_idx,pos])
            max_occ_length = max(max_occ_length, occ_length)

    n_d=occurences_list[0].shape[1]
    # Mélanger
    shuffle_indices = np.arange(len(occurences_list))
    random.shuffle(shuffle_indices)
    occurences_list = [occurences_list[i] for i in shuffle_indices]
    occurences_idxes_list = np.array(occurences_idxes_list)[shuffle_indices]

    all_motifs_concat = np.concatenate(occurences_list, axis=0)
    power_signal = np.mean(all_motifs_concat ** 2)

    noise_power_motifs = power_signal / (10 ** (snr_motifs_db / 10))
    noise_power_gaps = power_signal / (10 ** (snr_gaps_db / 10))

    sigma_motifs = np.sqrt(noise_power_motifs)
    sigma_gaps = np.sqrt(noise_power_gaps)

    # Construction du signal
    initial_gap = int(max_occ_length * sparsity + np.random.uniform(-1, 1) * sparsity_fluctuation)
    signal = np.random.normal(scale=sigma_gaps, size=(initial_gap, n_d))

    occurences_positions = {}
    for i, occ in enumerate(occurences_list):
        noisy_occ = occ + np.random.normal(scale=sigma_motifs, size=occ.shape)
        s1 = signal.shape[0]
        e1 = s1 + occ.shape[0]
        signal = np.concatenate((signal, noisy_occ), axis=0)

        label = occurences_idxes_list[i][0]
        occurences_positions.setdefault(label, []).append([s1, e1])

        gap_length = int(max_occ_length * sparsity + np.random.uniform(-1, 1) * sparsity_fluctuation)
        if interpolation == 'random':
            gap = np.random.normal(scale=sigma_gaps, size=(gap_length, n_d))
        elif interpolation == 'linear':
        # interpolation linéaire entre last_value et 0 sur gap_length points
            last_value = noisy_occ[-1]  # shape (n_d,)
            linear = np.linspace(last_value, np.zeros_like(last_value), gap_length)
            gap = linear + np.random.normal(scale=sigma_gaps, size=(gap_length, n_d))
        elif interpolation == 'spline': 
            last_value = noisy_occ[-1]  # shape (n_d,)
            spline = spline_random_control(last_value, np.zeros_like(last_value), gap_length)
            gap = spline + np.random.normal(scale=sigma_gaps, size=(gap_length, n_d))
        signal = np.concatenate((signal, gap), axis=0)

    occurences_mask = dict_to_mask(occurences_positions, signal.shape[0])
    
    
    return signal, occurences_mask, occurences_idxes_list.tolist()