#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:45:47 2018

@author: mi0240
"""

import os
import json
import pickle
from bs4 import BeautifulSoup as bs
from collections import Counter
import nltk.data
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

MIN_OCCUR =5 
MIN_RATIO= 0.05


def fashionTerms(json_data,termdict,keys):
    for key,values in json_data[keys].iteritems():
        termdict[key.encode('ascii','ignore')]= values
    return termdict

def load_terms(infile):
    with open(infile) as json_file:
        json_data=json.load(json_file)
    FBR ={}
    FSN ={}
    ART ={}
    ATTR ={}
    FBR = fashionTerms(json_data,FBR,'fabric')
    FSN = fashionTerms(json_data,FSN,'fashion_strings')
    ART = fashionTerms(json_data,ART,'article_types')
    ATTR = fashionTerms(json_data,ATTR,'attribute_types') 
    return FBR, FSN, ART, ATTR       

def replace_text(text):
    rep_chars=[ '/','%','|','@','=','*','+','[',']']
    text.replace('&quot;','"')
    text.replace("&apos;","'")
    text.replace("&nbsp;","")
    text.replace("\u2019","'")  
    text.replace("\u00a0","") 
    start= text.find('\u')
    while(start!=-1):
        end = text.find(' ',start)
        text = text.replace(text[start:end],'')
        start= text.find('\u')
    for char in rep_chars:
        text = text.replace(char,'')
    return text


def gloss_count_ratio(data, get_ratio = 0):
    term_count =0;
    word_count=0
    for key,values in allkeys.iteritems():
        if key in data :
            term_count+=1
    if (get_ratio):
        word_count += len(data.split())
        return term_count, float(term_count)/word_count
    else :
        return term_count


def append_sent(sent, train_sent, test_sent, test_keys):
    for lines in sent:
        
        l=lines
        #print l
        l = replace_text(l)
        #l = " ".join(l)
        if (len(l)>1):
            flag=0
            for k,v in test_keys.iteritems():
                if k in l.lower():
                    test_sent.append(l)
                    flag=1
                    break
            if flag==0 :
                train_sent.append(l)
    return train_sent, test_sent


fabric, fashion_strings, article_types, attribute_types= load_terms("../../../Trends_Glossary.json")
train_FBR, train_FSN, train_ART, train_ATTR = load_terms("train_terms.json")
test_FBR, test_FSN, test_ART, test_ATTR = load_terms("test_terms.json")
allkeys= dict ( fabric.items() + article_types.items() + fashion_strings.items() + attribute_types.items())
train_keys = dict(train_FBR.items() + train_FSN.items() + train_ART.items() + train_ATTR.items())
test_keys = dict(test_FBR.items() + test_FSN.items() + test_ART.items() + test_ATTR.items())

train_sent=[]
test_sent = []
file_names =[]
with open('../../../plot_csv_ngram.csv', 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields = csvreader.next()
 
    # extracting each data row one by one
    for row in csvreader:
        file_names.append(row[0])

lemmatizer = WordNetLemmatizer()
split_sent = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = set(stopwords.words('english'))

root = '../New Data/zalando.csv'
# for path, dirs, files in os.walk(root):
#     for name in files :.csv
#         with open (os.path.join(path, name)) as json_file:
#             print os.path.join(path, name)
#             raw_data = json.load(json_file)
#             try :
#                 unicode_content = raw_data['data']['bloglovin_content']['content']['content']
#                 data = replace_text(unicode_content)
#                 if len(data)==0 :
#                     continue
#                 term_count, ratio =gloss_count_ratio(data,1)
#                 if (MIN_OCCUR<term_count and MIN_RATIO<ratio):
#                     sent= split_sent.tokenize(unicode_content)#.encode('ascii','ignore'))
#                     train_sent, test_sent = append_sent(sent, train_sent, test_sent, test_keys)
#                     #print "In ['data']['bloglovin_content']['content']['content']"
#                     #append_sent(sent)
#             except :
#                 try:
#                     unicode_content = raw_data['data']['bloglovin_content']['content']
#                     data = replace_text(unicode_content)
#                     if len(unicode_content)==0 :
#                         continue
#                     term_count, ratio =gloss_count_ratio(data,1)
#                     if (MIN_OCCUR<term_count and MIN_RATIO< ratio):
#                         sent=split_sent.tokenize((data).encode('ascii','ignore') )
#                         train_sent, test_sent = append_sent(sent, train_sent, test_sent, test_keys)
#                         #print "in ['data']['bloglovin_content']['content']"
#                         #append_sent(sent)
#                 except :
#                     try:   
#                         unicode_content = raw_data['data']['mercury_content']['content'] 
#                     except : 
#                         continue
#                     soup = bs(unicode_content, 'html.parser')
#                     findp = soup.find_all('p') 
#                     gloss_count=0
#                     word_count=0
#                     paralist=[]
#                     for p in findp:
#                         tmp = replace_text(p.text)
#                         gloss_count += gloss_count_ratio(tmp, 0)
#                         word_count += len(tmp.split())
#                         if(len(tmp)>1) :
#                             paralist.append(tmp)
#                     if(word_count==0):
#                         continue
#                     gloss_ratio = float(gloss_count)/word_count
#                     #print "{},{},{}".format(gloss_ratio,gloss_count,word_count)
#                     if(MIN_RATIO<gloss_ratio and MIN_OCCUR<gloss_count):
#                         #print "In ['data']['mercury_content']['content']"
#                         for paras in paralist:    
#                             sent=split_sent.tokenize(paras.encode('ascii','ignore'))
#                             train_sent, test_sent = append_sent(sent, train_sent, test_sent, test_keys)
#                             #for lines in sent:
#                             #    allsent.append(lines)
#                 else :
#                     continue
#             else :
#                 continue

with open (root) as txt_file:
    sent = txt_file.readlines()
    train_sent, test_sent = append_sent(sent, train_sent, test_sent, test_keys)


import pickle
with open('train_sent_zalando.txt', 'w') as outfile:
    json.dump(train_sent, outfile, sort_keys=True, indent=4, separators=(',', ': '))
with open('test_sent_zalando.txt', 'w') as outfile:
    json.dump(test_sent, outfile, sort_keys=True, indent=4, separators=(',', ': '))
with open('train_100K_zalando.pkl', 'wb') as pklfile:
    pickle.dump(train_sent, pklfile, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_100K_zalando.pkl', 'wb') as pklfile:
    pickle.dump(test_sent, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
