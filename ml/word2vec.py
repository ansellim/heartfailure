import numpy as np
import pandas as pd

def generate_w2v_features():
    """
    Wrapper function to pull pickled data files for word embedding features
    """
    prefix = "../pre-processing/dataset_lemma_avg_v3/"

    w2v_features = []
    for l in ["train", "validation", "test"]:
        features = pd.read_pickle(prefix + f"dataset.lemma_avg_v3.seqs.{l}")
        w2v_features.append(np.array(features))
    
    return tuple(w2v_features)