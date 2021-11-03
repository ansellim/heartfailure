#manually curated keywords
#https://docs.google.com/spreadsheets/d/1J0Bvk7Vf9Q9R-lHoSDOdWhWtqTrLhJh5N5K7c-oCNjA/edit#gid=0

import re 
from fuzzywuzzy import fuzz

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

raw_data = pd.read_csv("bq-results-20211103-194424-9d3b9e96pe6j.csv")
hf_terms = pd.read_excel("heart_failure_terms.xlsx")

complete_terms = []
complete_terms_fuzzy = []
modifier_terms = []

def search_modifier_terms(text, hf_terms):
    terms = np.zeros(len(hf_terms))
    for t in range(len(hf_terms)):
        grps = re.findall(hf_terms[t], text)
        terms[t] = len(grps)
    
    return terms

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

for p in range(len(raw_data)):
    text = raw_data.loc[p, "clean_text"]

    ### search for complete terms
    complete_terms.append(search_complete_terms(text, hf_terms["term"].values))

    ### search for complete terms fuzzy
    complete_terms_fuzzy.append(search_complete_terms_fuzzy(text, hf_terms["term"].values))

    ### search for modifier terms
    modifier_terms.append(search_modifier_terms(text, hf_terms["term"].values))

complete_terms_features = np.array(complete_terms)
complete_terms_fuzzy_features = np.array(complete_terms_fuzzy)
modifier_terms_features = np.array(modifier_terms)

plt.figure(figsize=(20,15))
plt.imshow(complete_terms_features, interpolation='nearest', aspect='auto')
plt.colorbar()
plt.show()
plt.savefig("Sparse Matrix.png")
plt.close()

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
plt.show()
plt.savefig("Compare Sparse Matrix.png")
plt.close()

print("DONE")