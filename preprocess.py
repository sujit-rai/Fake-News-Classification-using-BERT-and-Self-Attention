import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))


glove_file = datapath('/home/sujit/Downloads/glove.twitter.27B/glove.twitter.27B.50d.txt')
word2vec_glove_file = get_tmpfile("glove.27B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

print(model["sep"])
input()


def preprocess(filename, savename):
	df = pd.read_csv(filename, sep="\t")
	df.label = df.label.map({"false":1, "barely-true":2, "pants-fire":0, "half-true":3, "mostly-true":4, "true":5})
	labels = df["label"].iloc[:].values
	df = df.fillna(" ")
	sentences = df["sentence"] + " [sep] " + df["justification"]
	sentences = sentences.values
	filtered_tokens = []
	glove_vect = []
	for inst in sentences:
		word_tokens = word_tokenize(inst)
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
		vectors = [model[w.lower()] for w in filtered_sentence if w in model.vocab]
		# filtered_tokens.append(filtered_sentence)
		glove_vect.append(vectors)
	np.save(savename+"_glove.npy", glove_vect)
	np.save(savename+"_labels.npy", labels)


preprocess("train.tsv", "train_multi_stop")
preprocess("test.tsv", "test_multi_stop")
preprocess("val.tsv", "val_multi_stop")


