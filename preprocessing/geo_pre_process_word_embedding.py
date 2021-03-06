# George Seah
# Requires gensim and fasttext
# Please download the clinical bert fast text word embedding here
# https://github.com/kexinhuang12345/clinicalBERT#gensim-word2vec-and-fasttext-models
# Daniel's data are pre-downloaded as csv and named as "hf_clinical_notes.csv"


nltk.download('stopwords')
nltk.download('wordnet')
import re
import string

import scipy
from sklearn.model_selection import train_test_split

# This will take 30 min++ to download
fasttext.util.download_model('en', if_exists='ignore')

# Update this portion to your local
m1 = KeyedVectors.load(r'C:\Users\JSEAH\CSE6250_Project\pre-processing\fasttext.model')
POSTIVE_DATA_PATH = r'C:\Users\JSEAH\CSE6250_Project\hf_positives.csv'
NEGATIVE_DATA_PATH = r'C:\Users\JSEAH\CSE6250_Project\hf_negative.csv'

# Loading the model can take some time
m2 = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(m2, 100)

def get_vector(word,m1,m2):
    if word in m1.wv.key_to_index:
        # Word found in fasttext
        w_ft_2_vec = m1.wv[word]
    else:
        # Word not found in fasttext
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
    non_essential_word_list =['admission','date','service','birth','also']
    new_text = re.sub('[0-9]{4}pm','',text) 
    new_text = ' '.join([i for i in new_text.split() if i not in non_essential_word_list])
    return new_text  

wnl = WordNetLemmatizer()

def get_pos( word ):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    most_common_pos_list = pos_counts.most_common(3)
    #print(most_common_pos_list[0][0])
    return most_common_pos_list[0][0]

def lemmatization(text):
    return wnl.lemmatize(text,get_pos(text))

def remove_numeric(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

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

def create_dataset(grouped_df,max_length,avg=False):
    #convert each note to word embedding
    seq_arrays = []
    for idx,note in enumerate(grouped_df['cleaned_text_2'].to_list()):
        if idx % 100==0:
            print(idx,"notes embedding processed")
        seq_arrays.append(note_to_vec(note))
    
    #pad it to same max_length x 100 matrix
    padded_seq_arrays=pad_seq_array(seq_arrays,max_length)
    if avg==True:
        #truncate to 4000 cover 95% of patient cases
        truncated_length = 4000
        padded_seq_arrays=[np.mean(i.toarray()[0:truncated_length],axis=0) for i in padded_seq_arrays]
    del seq_arrays[:]

    labels = grouped_df['hf_label'].to_list()
    patient_ids = grouped_df['SUBJECT_ID'].to_list()
    return patient_ids, labels, padded_seq_arrays


def split_test_train_dataset(positive_data_path,negative_data_path):
    positive_data = pd.read_csv(positive_data_path)
    positive_data['hf_label'] = 1
    negative_data = pd.read_csv(negative_data_path)
    negative_data['hf_label'] = 0
    data = positive_data.append(negative_data)

    # Clean up data
    data['cleaned_text_2'] = data['clean_text'].apply(remove_punctuations)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(lower_case)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(stopword_filter)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(Nchar_filter)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(remove_non_essential_words)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(lemmatization)
    data['cleaned_text_2'] = data['cleaned_text_2'].apply(remove_numeric)

    data = data.sort_values(by=['SUBJECT_ID', 'HADM_ID']).reset_index()

    # Concatenate discharge summaries by patient
    grouped_text = data.groupby('SUBJECT_ID')['cleaned_text_2'].apply(lambda x: ','.join(x)).reset_index()
    data_label = data[['SUBJECT_ID', 'hf_label']].drop_duplicates()

    # Create a combined dataframe of |SUBJECT_ID|notes|hf_label
    grouped_df = pd.merge(grouped_text, data_label, how='inner', on=['SUBJECT_ID'])
    max_length = max([len(note.split()) for note in grouped_df['cleaned_text_2'].to_list()])

    # Split the dataset into train, validation, test
    train_ratio = 0.60
    validation_ratio = 0.20
    test_ratio = 0.20

    train_data, val_and_test_data = train_test_split(grouped_df, test_size=1 - train_ratio, random_state=42)
    val_data, test_data = train_test_split(val_and_test_data, test_size=test_ratio / (test_ratio + validation_ratio),
                                           random_state=42)

    print(len(train_data), len(val_data), len(test_data), max_length)
    return train_data, val_data, test_data, max_length

