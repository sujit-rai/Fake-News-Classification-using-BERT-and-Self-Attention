import pandas as pd
import numpy as np

from sentistrength import PySentiStr
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

filepath = "/home/sujit/Downloads/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t')
emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()


# print(np.concatenate((model["sep"], emolex_words[emolex_words.word == 'charitable'].values[0][1:]), axis=0))
# input()

def preprocess(filename, metafilename, savename):
	df = pd.read_csv(filename, sep="\t")
	meta = pd.read_csv(metafilename, sep="\t")
	df.label = df.label.map({"false":0, "barely-true":0, "pants-fire":0, "half-true":1, "mostly-true":1, "true":1})
	labels = df["label"].iloc[:].values
	df = df.fillna(" ")
	sentences = df["sentence"] + " [sep] " + df["justification"]
	sentences = sentences.values
	filtered_tokens = []
	glove_vect = []
	index = 0
	for inst in sentences:
		mt = np.concatenate((meta.iloc[index].values, np.zeros((55,))), axis=0) - 0.5
		index = index + 1
		word_tokens = word_tokenize(inst)
		# filtered_sentence = [w for w in word_tokens if not w in stop_words]
		vectors = []
		for w in word_tokens:
			if w in model.vocab:
				gl = model[w.lower()]
				em = emolex_words[emolex_words.word == w.lower()].values
				if em.shape[0]==0:
					em = np.zeros((1,11))
				vectors.append(np.concatenate((gl, em[0][1:] - 0.5), axis=0))
		# vectors = [np.concatenate((model[w.lower()], emolex_words[emolex_words.word == w.lower()]), axis=0) for w in filtered_sentence if w in model.vocab]
		# filtered_tokens.append(filtered_sentence)
		vectors.append(mt)
		glove_vect.append(vectors)
	np.save(savename+"_glove_m.npy", glove_vect)
	np.save(savename+"_labels_m.npy", labels)


preprocess("train.tsv", "train2.tsv", "train_binary")
preprocess("test.tsv", "test2.tsv", "test_binary")
preprocess("val.tsv", "val2.tsv", "val_binary")


