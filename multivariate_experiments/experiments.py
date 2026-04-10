import numpy as np 
import pandas as pd
import os 
import json 
import time 
from utils import dict_to_mask, dict_to_dims, mask_to_dict, multivariate_to_univariate_labels

initial_path = os.getcwd()

os.chdir(initial_path+'/../')
from src.metric import EventScoreSubDims, EventScore, AdjustedMutualInfoScore



os.chdir(initial_path)

class Experiment: 

    def __init__(self,algorithms:list, thresholds = np.linspace(0,1,101),njobs=1,verbose = True) -> None:
   


        """Initialization

        Args:
            algorithms (list): list of algorithm classes
            
            thresholds (np.ndarray, optional): numpy array of thresholds to consider for the event based metric. Defaults to numpy.linspace(0,1,101).
        """
        self.algorithms = algorithms

        self.thresholds = thresholds
        self.njobs = njobs
        self.verbose = verbose


    def compute_scores_pepa(self, label, prediction):

        scores = []

        #event score
        lp,lr,lf = EventScore().score(label,prediction, self.thresholds)
        for t,p,r,f in zip(self.thresholds,lp,lr,lf): 
            scores.append([f"es-precision_{np.round(t,2)}",p])
            scores.append([f"es-recall_{np.round(t,2)}",r])
            scores.append([f"es-fscore_{np.round(t,2)}",f])
        scores.append(["es-auc-precision",np.mean(lp)])
        scores.append(["es-auc-recall",np.mean(lr)])
        scores.append(["es-auc-fscore",np.mean(lf)])
        scores.append(["amis",AdjustedMutualInfoScore().score(label,prediction)])

        return scores

    def compute_scores(self, label, prediction, true_dims, pred_dims): 

        scores = []

        #subdims event score
        lp,lr,lf,ldp,ldr,ldf = EventScoreSubDims().score(label,prediction,true_dims, pred_dims, self.thresholds)
        for t,p,r,f in zip(self.thresholds,lp,lr,lf): 
            scores.append([f"es-precision_{np.round(t,2)}",p])
            scores.append([f"es-recall_{np.round(t,2)}",r])
            scores.append([f"es-fscore_{np.round(t,2)}",f])

        scores.append(["es-auc-precision",np.mean(lp)])
        scores.append(["es-auc-recall",np.mean(lr)])
        scores.append(["es-auc-fscore",np.mean(lf)])
        scores.append(["es-dim-precision",np.mean(ldp)])
        scores.append(["es-dim-recall",np.mean(ldr)])
        scores.append(["es-dim-fscore",np.mean(ldf)])

        #ajusted mutual information
        scores.append(["amis",AdjustedMutualInfoScore().score(label,prediction)])

        return scores
    
    def compute_dim_by_dim_event_scores(self, true_labels_list, pred_labels_list):

        n_d = len(true_labels_list)
        scores = []

        # Listes pour agréger les valeurs (par threshold)
        lp_all, lr_all, lf_all = [], [], []

        # Boucle sur chaque dimension
        for d in range(n_d):
            try:
                lp, lr, lf = EventScore().score(
                    true_labels_list[d],
                    pred_labels_list[d],
                    self.thresholds
                )
                lp_all.append(lp)
                lr_all.append(lr)
                lf_all.append(lf)
            except Exception as e:
                print(f"⚠️ Error computing EventScore for dim {d}: {e}")
                # Ajoute des 0 si crash pour garder cohérence
                lp_all.append([0]*len(self.thresholds))
                lr_all.append([0]*len(self.thresholds))
                lf_all.append([0]*len(self.thresholds))

        # Moyenne sur les dimensions (moyenne par threshold)
        lp_mean = np.mean(lp_all, axis=0)
        lr_mean = np.mean(lr_all, axis=0)
        lf_mean = np.mean(lf_all, axis=0)

        lp_min = np.min(lp_all, axis=0)
        lr_min = np.min(lr_all, axis=0)
        lf_min = np.min(lf_all, axis=0)

        lp_max = np.max(lp_all, axis=0)
        lr_max = np.max(lr_all, axis=0)
        lf_max = np.max(lf_all, axis=0)

        # Stocke les scores individuels pour inspection éventuelle
        for t, p, r, f, p_min, r_min, f_min, p_max, r_max, f_max in zip(self.thresholds, lp_mean, lr_mean, lf_mean, lp_min, lr_min, lf_min, lp_max, lr_max, lf_max):
            scores.append([f"es-precision_{np.round(t, 2)}", p])
            scores.append([f"es-recall_{np.round(t, 2)}", r])
            scores.append([f"es-fscore_{np.round(t, 2)}", f])
            scores.append([f"es-precision_min_{np.round(t, 2)}", p_min])
            scores.append([f"es-recall_min_{np.round(t, 2)}", r_min])
            scores.append([f"es-fscore_min_{np.round(t, 2)}", f_min])
            scores.append([f"es-precision_max_{np.round(t, 2)}", p_max])
            scores.append([f"es-recall_max_{np.round(t, 2)}", r_max])
            scores.append([f"es-fscore_max_{np.round(t, 2)}", f_max])   
        # Scores AUC
        scores.append(["es-auc-precision", np.mean(lp_mean)])
        scores.append(["es-auc-recall", np.mean(lr_mean)])
        scores.append(["es-auc-fscore", np.mean(lf_mean)])


        return scores


    def run_experiment(
        self, dataset_path, configs: dict,
        threshold_type_list=None, ranking_method_list=None,
        results_path=None, verbose=True
    ):
        """
        Gère 3 cas :
        - BasePersistentPattern (PEPA) → pas de dimensions, EventScore uniquement
        - mPEPA → variations threshold + ranking
        - Autres algorithmes → compute_scores() complet (dims + time)
        """

        def write_scores_csv(folder, algo_name, config_name, i, scores):
            scores_df = pd.DataFrame(scores, columns=["metric", "value"])
            scores_df.to_csv(
                os.path.join(folder, f"scores_{algo_name}_{config_name}_{i}.csv"),
                index=False
            )

        def make_zero_scores(with_dims=True):
            """Scores = 0 si crash. Option pour désactiver les scores dimensionnels."""
            base = [["precision", 0], ["recall", 0], ["fscore", 0], ["time", 0]]
            dims = [["dim-precision", 0], ["dim-recall", 0], ["dim-fscore", 0]]
            return base + dims if with_dims else base

        def make_zero_scores_dimbydim():
            scores = []
            for t in self.thresholds:
                scores += [
                    [f"es-precision_{t}", 0],
                    [f"es-recall_{t}", 0],
                    [f"es-fscore_{t}", 0],
                    [f"es-precision_min_{t}", 0],
                    [f"es-recall_min_{t}", 0],
                    [f"es-fscore_min_{t}", 0],
                    [f"es-precision_max_{t}", 0],
                    [f"es-recall_max_{t}", 0],
                    [f"es-fscore_max_{t}", 0],
                ]
            scores += [
                ["es-auc-precision", 0],
                ["es-auc-recall", 0],
                ["es-auc-fscore", 0],
                ["amis", 0],
                ["time", 0],
            ]
            return scores
        
        self.dataset_path = dataset_path

        # ----------------------------------------------------------
        # 1) Création du dossier Results/
        # ----------------------------------------------------------
        self.results_path = results_path or os.path.join(self.dataset_path, "Results")
        os.makedirs(self.results_path, exist_ok=True)

        # ----------------------------------------------------------
        # 2) Chargement des données
        # ----------------------------------------------------------
        metadata_list = sorted([f for f in os.listdir(os.path.join(dataset_path, "Data")) if "metadata" in f])
        signal_list   = sorted([f for f in os.listdir(os.path.join(dataset_path, "Data")) if "signal" in f])
        n_signal = len(signal_list)

        # ----------------------------------------------------------
        # 3) Boucle sur les algorithmes enregistrés
        # ----------------------------------------------------------
        for algo in self.algorithms:
            algo_name = algo.__name__
            algo_configs = configs.get(algo_name, {})
            print(f"\n▶ Running {algo_name} on {n_signal} signals")

            for config_name, config_dict in algo_configs.items():

                base_config_folder = os.path.join(self.results_path, f"{algo_name}_{config_name}")
                os.makedirs(base_config_folder, exist_ok=True)

                config_to_save = {}
                for k, v in config_dict.items():
                    print(k,v)
                    if k=='univariate_method':
                        config_to_save[k] = v.__name__  # on garde juste le nom
                    else:
                        config_to_save[k] = v

                try:
                    with open(os.path.join(base_config_folder, "config.json"), "w") as f:
                        json.dump(config_to_save, f, indent=4)
                except Exception as e:
                    print(f"⚠️ Could not save config for {algo_name} {config_name}: {e}")
                    continue
                print(f"  → Config: {config_name}")

                # ----------------------------------------------------------
                # 4) Boucle sur les signaux
                # ----------------------------------------------------------
                for i in range(n_signal):

                    signal = pd.read_csv(os.path.join(dataset_path, "Data", signal_list[i])).to_numpy()
                    metadata = json.load(open(os.path.join(dataset_path, "Data", metadata_list[i])))

                    true_labels = dict_to_mask(metadata["occurences_positions"], signal.shape[0])
                    true_dims   = dict_to_dims(metadata["active_dims"])

                    model = algo(**config_dict)
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # ✅ CAS PARTICULIER : BasePersistentPattern (PEPA original)
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    if algo_name == "BasePersistentPattern" or algo_name == 'MatrixProfile' or algo_name == 'UnivariateAfterPCA':
                        try:
                            start = time.time()
                            model.fit(signal)
                            elapsed = time.time() - start

                            pred_labels = model.prediction_mask_

                            # Scores event–level (pas de dimensions)
                            scores = self.compute_scores_pepa(true_labels, pred_labels)
                            scores.append(["time", elapsed])
                            # JSON → uniquement occurences_positions
                            with open(os.path.join(base_config_folder, f"predictions_{i}.json"), "w") as f:
                                json.dump({
                                    "occurences_positions": mask_to_dict(pred_labels)
                                }, f, indent=4)


                        except Exception as e:
                            print(f"⚠️ ERROR (PEPA) on signal {i}, config={config_name} : {e}")
                            scores = make_zero_scores(with_dims=False)

                        write_scores_csv(base_config_folder, algo_name, config_name, i, scores)

            
                        print(f"    ✔ Finished signal {i}" f"algo={algo_name}, config={config_name}")
                        continue  # ⛔ Ne pas faire le flow normal

                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # ✅ CAS PARTICULIER mPEPA (DOIT RESTER COMME AVANT)
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    if algo_name == "mPEPA":

                        try:
                            start_base = time.time()
                            model.fit(signal)
                            base_fit_time = time.time() - start_base
                            print(f"      → Base fit OK ({base_fit_time:.3f}s)")
                        except Exception as e:
                            print(f"⚠️ Fit ERROR (base mPEPA) : {e}")
                            write_scores_csv(base_config_folder, algo_name, config_name, i, make_zero_scores())
                            continue

                        # variations (threshold_type × ranking method)
                        for th_type in threshold_type_list:
                            for rank_meth in ranking_method_list:

                                variation_name = f"{config_name}_{th_type}_{rank_meth}"
                                variation_folder = os.path.join(self.results_path, f"{algo_name}_{variation_name}")
                                os.makedirs(variation_folder, exist_ok=True)

                                try:
                                    start_var = time.time()
                                    model.set_threshold_type(th_type)
                                    model.set_post_process_method(rank_meth)

                                    pred_labels = model.prediction_mask_
                                    pred_dims = model.prediction_dimension_

                                    total_time = base_fit_time + (time.time() - start_var)

                                    scores = self.compute_scores(true_labels, pred_labels, true_dims, pred_dims)
                                    scores.append(["time", total_time])

                                    with open(os.path.join(variation_folder, f"predictions_{i}.json"), "w") as f:
                                        json.dump({
                                            "occurences_positions": mask_to_dict(pred_labels),
                                            "active_dims": {str(k): v for k, v in enumerate(pred_dims)}
                                        }, f, indent=4)

                                except Exception as e:
                                    print(f"⚠️ Variation ERROR ({variation_name}) : {e}")
                                    scores = make_zero_scores()

                                write_scores_csv(variation_folder, algo_name, variation_name, i, scores)

        

                        continue  # ⛔ important

                    if algo_name == "UnivariateDimByDim":
                        try:
                            start = time.time()
                            model.fit(signal)
                            elapsed = time.time() - start

                            estimated_masks_per_dim = model.prediction_mask_list_
                            # Conversion labels multivariés → univariés
                            true_labels_per_dim = multivariate_to_univariate_labels(
                                true_labels, true_dims, signal.shape[1]
                            )
                            # Calcul des scores
                            scores = self.compute_dim_by_dim_event_scores(
                                true_labels_per_dim,
                                estimated_masks_per_dim
                            )
                            scores.append(["time", elapsed])
                            with open(os.path.join(base_config_folder, f"predictions_{i}.json"), "w") as f:
                                json.dump({
                                    "occurences_positions_per_dim": [mask_to_dict(m) for m in estimated_masks_per_dim]
                                }, f, indent=4)

                        except Exception as e:
                            print(f"⚠️ ERROR (UnivariateDimByDim) on signal {i}, config={config_name} : {e}")
                            scores = make_zero_scores_dimbydim()

                        write_scores_csv(base_config_folder, algo_name, config_name, i, scores)
                        print(f"    ✔ Finished signal {i} (UnivariateDimByDim)")
                        continue

                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # ✅ CAS STANDARD 
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    try:
                        start = time.time()
                        model.fit(signal)
                        elapsed = time.time() - start

                        pred_labels = model.prediction_mask_
                        pred_dims = model.prediction_dimension_

                        scores = self.compute_scores(true_labels, pred_labels, true_dims, pred_dims)
                        scores.append(["time", elapsed])
                        with open(os.path.join(base_config_folder, f"predictions_{i}.json"), "w") as f:
                            json.dump({
                                "occurences_positions": mask_to_dict(pred_labels),
                                "active_dims": {str(k): v for k, v in enumerate(pred_dims)}
                            }, f, indent=4)
                    except Exception as e:
                        print(f"⚠️ Fit ERROR on signal {i}, config={config_name} : {e}")
                        scores = make_zero_scores()

                    write_scores_csv(base_config_folder, algo_name, config_name, i, scores)

            

                    print(f"    ✔ Finished signal {i}")

        #print("\n✅ All experiments completed!\n")
