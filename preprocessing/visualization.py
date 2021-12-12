# Ansel Lim

# dependencies

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize

# load dataset

train = pd.read_csv("./processed_data/train.csv")[['text', 'label']]
val = pd.read_csv("./processed_data/val.csv")[['text', 'label']]
test = pd.read_csv("./processed_data/test.csv")[['text', 'label']]
all_patients = pd.concat([train, val, test])
all_patients.reset_index(inplace=True)

# Show distribution of patient label (positive or negative) in the cohort

all_patients['label'] = all_patients['label'].replace(1, 'positive')
all_patients['label'] = all_patients['label'].replace(0, 'negative')
ax = sns.countplot(x="label", data=all_patients)
plt.title("Distribution of patients in the cohort")
plt.show()

# Show word counts for discharge summary (how long is each patient's discharge summary?)

all_patients['word_count'] = all_patients['text'].apply(lambda x: len(word_tokenize(x)))

ax = sns.histplot(data=all_patients, x='word_count')
plt.xlabel('Word count')
plt.title("Distribution of word count in the cohort of patients")
plt.show()


def remove_outliers(array):
    '''
    A function to remove outliers from a numpy array-like object.
    '''
    q3 = np.percentile(array, 75)
    q1 = np.percentile(array, 25)
    iqr = q3 - q1
    lower_threshold = q1 - 1.5 * iqr
    upper_threshold = q3 + 1.5 * iqr
    return [x for x in array if x >= lower_threshold and x <= upper_threshold]


# Show distribution of word count without outliers

ax = sns.histplot(remove_outliers(all_patients['word_count']))
plt.xlabel('Word count')
plt.title('Distribution of word count (without outliers)')
plt.show()
