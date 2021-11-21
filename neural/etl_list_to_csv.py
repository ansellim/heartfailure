# Previously, my data.pkl contained lists. Now, I convert the data into CSV format for easier access. The data is the same.

import pandas as pd
import pickle

with open("./datasets/data.pkl",'rb') as f:
    data = pickle.load(f)

train = pd.DataFrame({'text':data['train_texts'],'label':data['train_labels']})
val = pd.DataFrame({'text':data['val_texts'],'label':data['val_labels']})
test = pd.DataFrame({'text':data['test_texts'],'label':data['test_labels']})

train.to_csv("./datasets/train.csv")
val.to_csv("./datasets/val.csv")
test.to_csv("./datasets/test.csv")