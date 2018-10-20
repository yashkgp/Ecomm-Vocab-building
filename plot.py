import os
import json
import pickle
from bs4 import BeautifulSoup as bs
from collections import Counter
import nltk.data
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import csv
with open("terms_count_gloss.txt") as json_file:
    json_data = json.load(json_file)
gloss_all = list(json_data.keys())
gloss = [] #gloss with ngram  >= 2
for i in gloss_all :
	abc = len([word for word in i.split(" ")])
	if abc > 1 :
		gloss.append(i)
gloss_count = {}
ratio_ngram = {}
ratio_ngram['>.05'] = []
ratio_ngram['<.05'] = []
for i,x in enumerate(gloss):
	gloss_count[x.lower()]=0
	gloss[i] = x.lower()
ratio_csv = []
def intersection(lst1, lst2):
	counter = 0
	for i in (lst2):
		tmp=lst1.find(i)
		if (tmp!= -1 ):
			gloss_count[i]+=1
			counter+=1
	return counter
def  gloss_in_text(file_path,name):
	file_stat = []
	file_stat.append(name)
	with open(file_path) as f:
   		words = [word.lower() for line in f for word in line.split()]
   	sen = (" ").join(words)
	doc_terms = len(words)
	gloss_terms = intersection(sen,gloss)
	if (doc_terms!= 0):
		ratio = float(gloss_terms)/float(doc_terms)
	else :
		ratio = 0
	file_stat.append(doc_terms)
	file_stat.append(gloss_terms)
	file_stat.append(ratio)
	if (ratio >=0.05):
		ratio_ngram['>.05'].append(name)
		ratio_csv.append(file_stat)
	else :
		ratio_ngram['<.05'].append(name)


data_cleaned = os.getcwd() + '/data_cleaned'
for path, dirs, files in os.walk(data_cleaned):
    for name in tqdm(files) :
        file_path = (os.path.join(path, name))
        gloss_in_text(file_path,name)

with open("plot_csv_ngram.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ratio_csv)
with open('gloss_count_ngram.txt', 'w') as outfile:
    json.dump(gloss_count, outfile, sort_keys=True, indent=4, separators=(',', ': '))
with open('ratio>0.5.txt', 'w') as outfile:
    json.dump(ratio_ngram, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    	