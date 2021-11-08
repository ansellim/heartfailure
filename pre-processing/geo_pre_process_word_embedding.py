#Requires gensim and fasttext
#Please download the clinical bert fast text word embedding here
#https://github.com/kexinhuang12345/clinicalBERT#gensim-word2vec-and-fasttext-models
#Daniel's data are pre-downloaded as csv and named as "hf_clinical_notes.csv"



from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re, string
from sklearn.model_selection import train_test_split
import scipy

#this will take 30 min++ to download
fasttext.util.download_model('en', if_exists='ignore')  # English

#Update this portion to your local 
m1 = KeyedVectors.load(r'C:\Users\JSEAH\CSE6250_Project\pre-processing\fasttext.model')
POSTIVE_DATA_PATH  = r'C:\Users\JSEAH\CSE6250_Project\hf_positives.csv'
NEGATIVE_DATA_PATH = r'C:\Users\JSEAH\CSE6250_Project\hf_negative.csv'

#loading the model can take some time
m2 = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(m2, 100)

def get_vector(word,m1,m2):
    if word in m1.wv.key_to_index:
        w_ft_2_vec = m1.wv[word]

    else:
        #print("word not found in clinical note FastText")
        w_ft_2_vec = m2.get_word_vector(word)
    return w_ft_2_vec

def remove_punctuations(text):
    # remove punctuation and enter
    new_text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    new_text = new_text.replace('\n', ' ')
    return new_text

def lower_case(text):
    return text.lower()

def stopword_filter(text):
    stop = stopwords.words('english')
    return ' '.join([word for word in text.split() if word not in (stop)])
                    
def Nchar_filter(text):
    Value = ' '.join([i for i in text.split() if len(i) >= 3])
    return Value

def remove_non_essential_words(text):
    non_essential_word_list =['admission date','service','date birth']
    new_text = re.sub('[0-9]{4}pm','',text) 
    new_text = ' '.join([i for i in new_text.split() if i not in non_essential_word_list])
    return new_text  


### next step:
### ###
## Add the method to convert clinical note to word embedding array
### ###


def note_to_vec(input_note):
    """
    Input: input_notes in each row
    Return: word vector
    """
    #1 break input notes into text list
    #2 convert each text into a vector
    #Append them into a matrix of n x 100 ,where n = number of words in input_notes
    word_list = [get_vector(i,m1,m2) for i in input_note.split()]
    return np.array(word_list)

def pad_seq_array(seq_arrays,max_length):
    #print(max_length)
    new_arrays = []
    for idx,seq in enumerate(seq_arrays):
        if idx %100 == 0:
            print(idx, " note padding processed ")
        length = len(seq)
        try:
            zero_matrix = np.zeros((max_length-length, 100))
            new_seq = np.vstack((seq,zero_matrix))
            new_seq_csr_matrix = scipy.sparse.csr_matrix(new_seq)
        except:
            print("seq length is ",length,max_length)
        new_arrays.append(new_seq_csr_matrix)
    return new_arrays

def create_dataset(grouped_df,max_length):
    #convert each note to word embedding
    seq_arrays = []
    for idx,note in enumerate(grouped_df['cleaned_text_2'].to_list()):
        if idx % 100==0:
            print(idx,"notes embedding processed")
        seq_arrays.append(note_to_vec(note))
    
    #pad it to same max_length x 100 matrix
    padded_seq_arrays=pad_seq_array(seq_arrays,max_length)
    del seq_arrays[:]
    labels = grouped_df['hf_label'].to_list()
    patient_ids = grouped_df['SUBJECT_ID'].to_list()
    return patient_ids, labels, padded_seq_arrays


def split_test_train_dataset(positive_data_path,negative_data_path):
    positive_data = pd.read_csv(positive_data_path)
    positive_data['hf_label']=1
    negative_data = pd.read_csv(negative_data_path)
    negative_data['hf_label']=0
    data = positive_data.append(negative_data)
    #clean up
    data['cleaned_text_2'] = data['clean_text'].apply(remove_punctuations)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(lower_case)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(stopword_filter)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(Nchar_filter)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(remove_non_essential_words)
    data = data.sort_values(by=['SUBJECT_ID', 'HADM_ID']).reset_index()
    #group multiple discharge summary into single notes
    grouped_text = data.groupby('SUBJECT_ID')['cleaned_text_2'].apply(lambda x: ','.join(x)).reset_index()
    data_label = data[['SUBJECT_ID','hf_label']].drop_duplicates()
    #create a grouped_df of |SUBJECT_ID|notes|hf_label
    grouped_df = pd.merge(grouped_text,data_label,how='inner',on=['SUBJECT_ID'])
    max_length=max([len(note.split()) for note in grouped_df['cleaned_text_2'].to_list()])

    #split the dataset into train, test
    train_data,test_data = train_test_split(grouped_df,test_size=0.2,random_state = 123)
    print(len(train_data),len(test_data),max_length)
    return train_data,test_data,max_length



if __name__ == "__main__":
    train_ds,test_ds,max_length = split_test_train_dataset(POSTIVE_DATA_PATH,NEGATIVE_DATA_PATH)
    print("train test data set split completed")

    train_ids, train_labels, train_seqs = create_dataset(train_ds,max_length)
    pickle.dump(train_ids, open("dataset.ids.train", 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open("dataset.labels.train", 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_seqs, open("dataset.seqs.train", 'wb'), pickle.HIGHEST_PROTOCOL)
    print("train  data set created")


    test_ids, test_labels, test_seqs = create_dataset(test_ds,max_length)
    pickle.dump(train_ids, open("dataset.ids.test", 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open("dataset.labels.test", 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_seqs, open("dataset.seqs.test", 'wb'), pickle.HIGHEST_PROTOCOL)
    print("test  data set created")



    ### to read the data    
    #train_seqs = pickle.load(open('dataset.seqs.train'  , 'rb'))
    #train_seqs[0].toarray().shape  # use toarray() to convert back to numpy array