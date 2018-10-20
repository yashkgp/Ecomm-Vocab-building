import pickle
import json
from nltk import pos_tag, word_tokenize
import pandas as pd
import numpy as np
import os
import pickle
import json
import sklearn_crfsuite  
from  sklearn_crfsuite import metrics
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def load_pkl_file(infile):
    with open(infile) as pklfile:
        sent = [line.decode('utf-8').strip() for line in pklfile.readlines()]
    return sent
if __name__ == '__main__':
	count = 0
	bigrams = 0
	test_sent = load_pkl_file ('netaporter_new.csv') 
	for line in test_sent:
		for i,char in enumerate(line):
			prev = 0
			if (ord(char)>= ord('A') and ord(char) <= ord('Z')):
				count+=1
				tmpstr = line[prev:i+1]
				if (tmpstr.count(' ')>=1 and tmpstr.count(' ')<3):
					bigrams+=1
				prev = i 

	print "bigrams:" +str(bigrams)
	print " N-grams:"+str(count)


