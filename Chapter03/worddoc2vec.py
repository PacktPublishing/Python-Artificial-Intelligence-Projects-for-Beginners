
# coding: utf-8

# In[1]: import gensim untuk doc2vec dan logging untuk opsi informasi


import gensim,logging


# In[2]: setting setiap aktifitas keluar log nya yang berisi info waktu, level,pesan
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[3]: instansiasi word2vec dari google model for negative vectors  


gmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# In[4]: testing gmodel untuk cat dog dan spatula
gmodel['cat']
# In[4]: dog

gmodel['dog']
# In[4]: spatula
gmodel['spatula']

# In[5]: mengecek similaritas

gmodel.similarity('cat','dog')
# In[5]:

gmodel.similarity('cat','spatula')

# In[6]: import modul tagged doc dan doc2vec


from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


# In[7]: buat fungsi buat menghapus html tags dan perkocakan dunia
import re
def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) #hapus tag html
    sent = re.sub(r'(\w)\'(\w)', ' ', sent) #hapus petik satu
    sent = re.sub(r'\W', ' ', sent) #hapus tanda baca
    sent = re.sub(r'\s+', ' ', sent) #hapus spasi yang berurutan
    return sent.split()


import random
class PermuteSentences(object):
    def __init__(self,sents):
        self.sents = sents
        
    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent

# In[8]: unsupervised training data



import os
unsup_sentences = []

for dirname in ["train/pos","train/neg","train/unsup","test/pos","test/neg"]:
    for fname in sorted(os.listdir("aclImdb/"+dirname)):
        if fname[-4:] == '.txt':
            with open("aclImdb/"+dirname+"/"+fname,encoding='UTF-8') as f:
                sent = f.read()
                words = extract_words(sent)
                unsup_sentences.append(TaggedDocument(words,[dirname+"/"+fname]))

# In[9]: data ke dua
                
for dirname in ["review_polarity/txt_sentoken/pos","review_polarity/txt_sentoken/neg"]:
    for fname in sorted(os.listdir(dirname)):
        if fname[-4:] == '.txt':
            with open(dirname+"/"+fname,encoding='UTF-8') as f:
                for i, sent in enumerate(f):
                    words = extract_words(sent)
                    unsup_sentences.append(TaggedDocument(words,["%s/%s-%d" % (dirname,fname,i)]))

# In[10]: data ketiga
with open("stanfordSentimentTreebank/original_rt_snippets.txt",encoding='UTF-8') as f:
    for i, sent in enumerate(f):
        words = extract_words(sent)
        unsup_sentences.append(TaggedDocument(words,["rt-%d" % i]))



# In[8]: lihat banyaknya isi unsup_sentences
        
len(unsup_sentences)
# In[8]: 10 data pertama
unsup_sentences[0:1]



# In[9]: kocok dulu mang sebelum jadiiin model

permuter = PermuteSentences(unsup_sentences)

# In[9]: baru dibuat modelnya
model = Doc2Vec(permuter, dm=0,hs=1,vector_size=52)

# In[10]: kosongin memeri dulu gaes biar enteng

model.delete_temporary_training_data(keep_inference=True)

# In[11]: modelnya bisa di simpan jadi file lho

model.save('reviews.d2v')

# In[111]: load mpodel dari file
model = Doc2Vec.load('reviews.d2v')

# In[12]: infer

model.infer_vector(extract_words("This place is not worth your time"))

# In[13]: Mengecek similaritas


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))],
        [model.infer_vector(extract_words("Service sucks."))])

# In[13]: Mengecek similaritas 2


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(extract_words("Highly recommended."))],
        [model.infer_vector(extract_words("Service sucks."))])


# In[19]: load real dataset untuk prediksi


sentvecs = []
sentences = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]:
    with open("sentiment labelled sentences/%s_labelled.txt" % fname, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split('\t')
            sentences.append(line_split[0])
            words = extract_words(line_split[0])
            sentvecs.append(model.infer_vector(words, steps=10)) #buat vektor dari dokumen ini
            sentiments.append(int(line_split[1]))


# In[21]: kocok, vektorisasi, sentiment bersama

combined = list(zip(sentences, sentvecs, sentiments))
random.shuffle(combined)
sentences, sentvecs, sentiments = zip(*combined)


# In[37]: instansiasi KNN dan RF


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

clf = KNeighborsClassifier(n_neighbors=9)
clfrf = RandomForestClassifier()


# In[38]: cek skore KNN

scores = cross_val_score(clf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)

# In[43]: skor RF

scores = cross_val_score(clfrf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)


# In[44]: lakukan skoring dari vektorisasi,tfidf dan rf buat bandingin

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
pipeline = make_pipeline(CountVectorizer(),TfidfTransformer(), RandomForestClassifier())

scores = cross_val_score(pipeline,sentences, sentiments, cv=5)
np.mean(scores), np.std(scores)
