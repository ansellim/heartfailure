#manually curated keywords
#https://docs.google.com/spreadsheets/d/1J0Bvk7Vf9Q9R-lHoSDOdWhWtqTrLhJh5N5K7c-oCNjA/edit#gid=0

import re
from datetime import datetime

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def search_complete_terms(text, hf_terms):
    terms = np.zeros(len(hf_terms))
    for t in range(len(hf_terms)):
        grps = re.findall(hf_terms[t], text)
        terms[t] = len(grps)

    return terms

def search_complete_terms_fuzzy(text, hf_terms):
    terms = np.zeros(len(hf_terms))
    split_text = list(filter(lambda x: x!="", re.split("\s", re.sub("[^A-Za-z0-9(\\n)(\\s)]+", "", text))))

    for t in range(len(hf_terms)):
        q_term = hf_terms[t]
        n = len(q_term.split())

        text_combis = [" ".join(split_text[i: i + n]) for i in range(0,len(split_text), n)]
        text_ratios = [(t, fuzz.ratio(t, q_term)) for t in text_combis]
        similar_text = list(filter(lambda x: x[1] >= 95, text_ratios))

        terms[t] = len(similar_text)
    
    return terms

def search_modifier_terms(text, hf_terms):
    terms = np.zeros(len(hf_terms))
    for t in range(len(hf_terms)):
        grps = re.findall(hf_terms[t], text)
        terms[t] = len(grps)
    
    return terms

def get_elapsed_duration(data):

    admit = data["text"].apply(lambda x: re.search("Admission Date:\s+\[\*\*\d+-\d+-\d+\*\*\]", x))
    discharge = data["text"].apply(lambda x: re.search("Discharge Date:\s+\[\*\*\d+-\d+-\d+\*\*\]", x))
    
    admit = admit.apply(lambda x: x.group() if x != None else "1999-01-01") 
    discharge = discharge.apply(lambda x: x.group() if x != None else "1999-01-01") 
    
    admit = admit.apply(lambda x: datetime.strptime(re.search("\d+-\d+-\d+", x).group(), "%Y-%m-%d").date())
    discharge = discharge.apply(lambda x: datetime.strptime(re.search("\d+-\d+-\d+", x).group(), "%Y-%m-%d").date())

    duration = pd.concat([discharge, admit], keys =["discharge", "admit"], axis = 1)
    duration["elapsed"] = (duration["discharge"] - duration["admit"]).apply(lambda x: x.days).abs()

    q_date = datetime.strptime("1999-01-01", "%Y-%m-%d").date()
    duration["admit_count"] = duration["admit"].apply(lambda x : 1 if x == q_date else 0)
    duration["discharge_count"] = duration["discharge"].apply(lambda x : 1 if x == q_date else 0)
    duration["total_count"] = duration["admit_count"] + duration["discharge_count"]

    duration.loc[duration["total_count"] == 1, "elapsed"] = 0

    return duration["elapsed"].values[:, np.newaxis]

def min_max_norm(mat):
    scaler = MinMaxScaler()
    scaler.fit(mat)
    return scaler.transform(mat)

def plot_features(full_features):
    plt.figure(figsize=(20,15))
    plt.imshow(full_features, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.savefig("Full Features Sparse Matrix.png")
    plt.close()

def compare_features_plot(complete_terms_features, complete_terms_fuzzy_features, modifier_terms_features):
    fig, axs = plt.subplots(3, figsize = (20,15))
    ax0 = axs[0].imshow(complete_terms_features, interpolation='nearest', aspect='auto')
    axs[0].set_title('Complete terms')
    fig.colorbar(ax0, ax = axs[0])
    ax1 = axs[1].imshow(complete_terms_fuzzy_features, interpolation='nearest', aspect='auto')
    axs[1].set_title('Complete terms Fuzzy')
    fig.colorbar(ax1, ax = axs[1])
    ax2 = axs[2].imshow(modifier_terms_features, interpolation='nearest', aspect='auto')
    axs[2].set_title('Modifier terms')
    fig.colorbar(ax2, ax = axs[2])
    plt.savefig("Compare Sparse Matrix.png")
    plt.close()

if __name__ == "__main__":
    # raw_negative = pd.read_csv("bigquery/bq_data_in_csv/hf_negative.csv")
    # raw_positive = pd.read_csv("bigquery/bq_data_in_csv/hf_positives.csv")

    # labels = np.vstack([np.zeros((len(raw_negative),1)), np.ones((len(raw_positive), 1))])

    # raw_data = pd.concat([raw_negative, raw_positive], axis=0, sort=False).reset_index(drop =True)

    data = pd.read_pickle("ml/datasets/data.pkl")
    hf_terms = pd.read_excel("heart_failure_terms.xlsx")

    processed_data = {}
    for grp in ["train", "valid", "test"]: 
        raw_data = data[grp + "_texts"].copy()
        # elapsed = get_elapsed_duration(raw_data)

        complete_terms = []
        # complete_terms_fuzzy = []
        modifier_terms = []

        for p in range(len(raw_data)):
            text = raw_data[p]

            ### search for complete terms
            complete_terms.append(search_complete_terms(text, hf_terms["term"].values))

            ### search for complete terms fuzzy
            # complete_terms_fuzzy.append(search_complete_terms_fuzzy(text, hf_terms["term"].values))

            ### search for modifier terms
            modifier_terms.append(search_modifier_terms(text, hf_terms["term"].values))

        complete_terms_features = np.array(complete_terms)
        # complete_terms_fuzzy_features = np.array(complete_terms_fuzzy)
        modifier_terms_features = np.array(modifier_terms)

        # full_features = np.hstack([elapsed, complete_terms_features])
        full_features = min_max_norm(complete_terms_features)

        plot_features(full_features)
        # compare_features_plot(complete_terms_features, complete_terms_fuzzy_features, modifier_terms_features)

        processed_data[grp + "_texts"] = full_features.copy()

        processed_data[grp + "_labels"] = data[grp + "_labels"].copy()

    