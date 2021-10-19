#Requires gensim and fasttext
#Please download the clinical bert fast text word embedding here
#https://github.com/kexinhuang12345/clinicalBERT#gensim-word2vec-and-fasttext-models

from gensim.models import FastText,KeyedVectors
from gensim.models.fasttext import load_facebook_model
import fasttext
import fasttext.util

#this will take 30 min++ to download
fasttext.util.download_model('en', if_exists='ignore')  # English

m1 = KeyedVectors.load(r'C:\Users\JSEAH\BDHI\docker_shared\project\fasttext.model')


#loading the model can take some time
m2 = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(m2, 100)

def get_vector(word,m1,m2):
    if word in m1.wv.key_to_index:
        w_ft_2_vec = m1.wv[word]

    else:
        print("word not found in clinical note FastText")
        w_ft_2_vec = m2.get_word_vector("epithemia")
    return w_ft_2_vec


### next step:
### ###
## Add the method to convert clinical note to word embedding pytorch layer
### ###

if __name__ == "__main__":
    print("heart word embedding vector",get_vector("heart",m1,m2))
    print("failure word embedding vector",get_vector("failure",m1,m2))
    print("epithemia out-of-vocab embedding vector",get_vector("epithemia",m1,m2))
