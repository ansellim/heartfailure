import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim

from plots import plot_confusion_matrix,plot_learning_curves
from utils import train, evaluate
from model import MyVariableRNN
from datetime import datetime ## myaddition
import numpy as np #myaddition
import matplotlib.pyplot as plt #myaddition
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "../pre-processing/dataset.lemma_v2.seqs.train"
PATH_TRAIN_LABELS = "../pre-processing/dataset.lemma_v2.labels.train"
PATH_VALID_SEQS =  "../pre-processing/dataset.lemma_v2.seqs.validation"
PATH_VALID_LABELS =  "../pre-processing/dataset.lemma_v2.labels.validation"
PATH_TEST_SEQS = "../pre-processing/dataset.lemma_v2.seqs.test"
PATH_TEST_LABELS = "../pre-processing/dataset.lemma_v2.labels.test"
PATH_TEST_IDS = "../pre-processing/dataset.lemma_v2.ids.test"
PATH_OUTPUT = "../output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)



NUM_EPOCHS = 15
BATCH_SIZE = 64
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0
MAX_LENGTH=2500

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("PARAMETER",device,NUM_EPOCHS,BATCH_SIZE,MAX_LENGTH)

torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))

#truncate by max_len
train_seqs = [i[0:MAX_LENGTH] for i in train_seqs]
train_labels = train_labels[0:MAX_LENGTH]
print("train data length",len(train_seqs),train_seqs[0].shape)

valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))


#truncate by max_len
valid_seqs = [i[0:MAX_LENGTH] for i in valid_seqs]
valid_labels = valid_labels[0:MAX_LENGTH]
test_seqs = [i[0:MAX_LENGTH] for i in test_seqs]
test_labels = test_labels[0:MAX_LENGTH]

#print(train_seqs) #myaddition

num_features = 100

# not using ..use standard one
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        seqs_arrays = self.encodings[idx].toarray()
        labels_arrays = self.labels[idx]
        return seqs_arrays,labels_arrays

    def __len__(self):
        return len(self.labels)

def notes_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	I have padded the matrix to fix size 30768 x 100 in pre-processing . hence don't need to do any additional padding here

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""
	seqs_array = np.array([x[0] for x in batch])
	labels_array = np.array([x[1] for x in batch])

	seqs_tensor = torch.FloatTensor(seqs_array)
	labels_tensor = torch.LongTensor(labels_array)

	return seqs_tensor, labels_tensor


#Create Tensor Dataset
#testing code to use default tensor to load, 
#but it is not efficient to memory and time since need to load the entiredataset twice
#train_labels = torch.LongTensor(train_labels)
#train_seqs = torch.FloatTensor(train_seqs)
#test_labels = torch.LongTensor(test_labels)
#test_seqs = torch.FloatTensor(test_seqs)
#valid_labels = torch.LongTensor(valid_labels)
#valid_seqs = torch.FloatTensor(valid_seqs)
#print("Data loaded as Tensor")

#train_dataset = TensorDataset(train_seqs,train_labels)
#valid_dataset = TensorDataset(valid_seqs,valid_labels)
#test_dataset = TensorDataset(test_seqs,test_labels)

train_dataset = MyDataset(train_seqs,train_labels)
valid_dataset = MyDataset(valid_seqs,valid_labels)
test_dataset = MyDataset(test_seqs,test_labels)




#remove collate function as I already padded the data into same size
#collate_fn = notes_collate_fn
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn = notes_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn = notes_collate_fn, num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,collate_fn = notes_collate_fn, num_workers=NUM_WORKERS)

model = MyVariableRNN(num_features)
print("Model Parameter ",model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
test_losses, test_accuracies = [],[]
for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)
	


	train_losses.append(train_loss)
	valid_losses.append(valid_loss)
	#test_losses.append(test_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)
	#test_accuracies.append(test_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		#test_acc_best_model = test_accuracy
		#test_results_best_model = test_results
		torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"),
				   _use_new_zipfile_serialization=False # TO AVOID PROBLEMS WITH LOADING THE PTH FILE ON GRADESCOPE? APPARENTLY IT'S A PYTORCH VERSION PROBLEM?
				   )

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))
# TODO: For your report, try to make plots similar to those in the previous task.
# TODO: You may use the validation set in case you cannot use the test set.


plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

def compute_metrics(results):
	y_true = [i[0] for i in results]
	y_pred = [i[1] for i in results]
	acc = accuracy_score(y_true, y_pred)
	auc = roc_auc_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	return {'accuracy': acc,'auc': auc,'precision': precision,'recall': recall,'f1_score': f1}

# TODO: Complete predict_mortality
def predict_mortality(model, device, data_loader):
	model.eval()
	# TODO: Evaluate the data (from data_loader) using model,
	# TODO: return a List of probabilities
	#probas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	probas = []
	for batch, label in test_loader:
		model = model.to(device)
		y = model(batch)
		probabilities = torch.sigmoid(y)
		#print(probabilities[0][1].item())
		prob = probabilities[0][1].item()
		probas.append(prob)
	return probas

#test_prob = predict_mortality(best_model, device, test_loader)


test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion,print_freq=500)
plot_confusion_matrix(test_results,['0','1'])
metrics = compute_metrics(test_results)
print(metrics)
print('test accuracy ',test_accuracy,test_loss)
print(test_results)

