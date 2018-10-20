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
with open("gloss_count_ngram.txt") as json_file:
    gloss_data = json.load(json_file)
with open("terms_count_gloss2.txt") as json_file:
    json_data = json.load(json_file)
fabric = dict((k.lower(), v.lower()) for k,v in json_data['fabric'].iteritems())
article_types = dict((k.lower(), v.lower()) for k,v in json_data['article_types'].iteritems())
fashion_strings = dict((k.lower(), v.lower()) for k,v in json_data['fashion_strings'].iteritems())
attribute_types = dict((k.lower(), v.lower()) for k,v in json_data['attribute_types'].iteritems())

gloss_formated = {}
gloss_formated['fabric']= {}
gloss_formated['article_types']= {}
gloss_formated['fashion_strings']= {}
gloss_formated['attribute_types']= {}
for key, value in gloss_data.iteritems():
	if (value!= 0):
		if key in fabric :
			tmp ={}
			tmp[key] = value
			gloss_formated['fabric'][key] = value
		elif key in article_types:
			tmp ={}
			tmp[key] = value
			gloss_formated['article_types'][key] = value
		elif (key in fashion_strings):
			tmp ={}
			tmp[key] = value
			gloss_formated['fashion_strings'][key] =value
		elif (key in attribute_types):
			tmp ={}
			tmp[key] = value
			gloss_formated['attribute_types'][key] =value

with open('gloss_formatted_ngrams.txt', 'w') as outfile:
    json.dump(gloss_formated, outfile, sort_keys=True, indent=4, separators=(',', ': '))

