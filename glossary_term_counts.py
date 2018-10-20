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
from tqdm import tqdm

MIN_OCCUR =5 
MIN_RATIO= 0.05
recalldict=Counter()

def fashionTerms(termdict,keys):
    char_to_replace =  ['\u00e9', '\u2122', '\u1ebf', '\u00c2','\u00ae']
    for items in keys:
        getItem=items.replace('_',' ')
        #items.encode('ascii','ignore')
        for chars in char_to_replace:
            getItem=getItem.replace(chars,"")
        getItem=getItem.replace("\\","").strip()
        termdict[getItem]=getItem
    return termdict
 

# Disjoint set is required so do set difference
# dicta-dictb if flag==0
def removeCommon(dicta, dictb,flag=0) : 
    keys_a = set(dicta.keys())
    keys_b = set(dictb.keys())
    intersection = keys_a & keys_b 
    #print intersection
    if flag==0 :
        for i in intersection :
            del dicta[i]
    else :
        for i in intersection :
            del dictb[i]

def get_terms_in_classes(json_data):
    brands ={} 
    fabric ={}
    article_types={}
    fashion_strings={}
    attribute_types={}
    attribute_values={}
    brands = fashionTerms(brands,json_data['brands'])
    fabric = fashionTerms(fabric,json_data['fabric'])
    fashion_strings = fashionTerms(fashion_strings,json_data['fashion_strings'])
    article_types = fashionTerms(article_types,json_data['article_types'])
    attribute_types = fashionTerms(attribute_types,json_data['attribute_types'])
    attribute_values = fashionTerms(attribute_values,json_data['attribute_values'])
    attribute_types.update(attribute_values)
    del attribute_values
    #return fabric, fashion_strings, article_types, attribute_types
    #create disjoint set
    removeCommon(brands,fabric)
    removeCommon(article_types,brands)
    removeCommon(article_types,fabric)
    removeCommon(attribute_types,brands)
    removeCommon(attribute_types,fabric)
    removeCommon(attribute_types,article_types)
    removeCommon(fashion_strings,brands)
    removeCommon(fashion_strings,fabric)
    removeCommon(fashion_strings,article_types)
    removeCommon(fashion_strings,attribute_types)
    return fabric, fashion_strings, article_types, attribute_types

def delete_amiguous_keys(termdict):
    delKeys=['w','g','and','on','ant', 'na','ag','id','24','trends',
            'only','even','with','new','no','thin','make','end']
    for i in delKeys :
        try :
            del termdict[i]
            #print i
        except :
            pass
    return termdict

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
            #Found term. So increase its count
            recalldict[key]+=1
            term_count+=1
    if (get_ratio):
        word_count += len(data.split())
        #print "{},{},{}".format(float(term_count)/word_count,term_count,word_count)
        return float(term_count)/word_count
    else :
        return term_count


def append_sent(sent):
    for lines in sent:
        l=lines
        l = replace_text(l)
        l=l.strip()
        if (len(l)>1):
           allsent.append(l)

def setToDict(S):
    return {e:e for e in S}

def count_terms(recalldict,lists):
    tmp={}
    for items in lists:
        tmp[items]=recalldict[items]
    return tmp

with open("Trends_Glossary.json") as json_file:
    json_data = json.load(json_file)

fabric, fashion_strings, article_types, attribute_types = get_terms_in_classes(json_data)
print ("fabric = {}".format(len(fabric)))
print ("article_types = {}".format(len(article_types)))
print ("fashion_strings = {}".format(len(fashion_strings)))
print ("attribute_types = {}".format(len(attribute_types)))
allkeys= dict ( fabric.items() + article_types.items() + fashion_strings.items() + attribute_types.items())
allkeys = delete_amiguous_keys(allkeys)
allsent = {}
allsent['fabric'] = fabric
allsent['article_types'] = article_types
allsent['fashion_strings']  = fashion_strings 
allsent['attribute_types'] = attribute_types

# lemmatizer = WordNetLemmatizer()
# split_sent = nltk.data.load('tokenizers/punkt/english.pickle')
# stop_words = set(stopwords.words('english'))
# data_cleaned = os.getcwd() + '/data_cleaned'
# root = os.getcwd()+ '/data100K'
# for path, dirs, files in os.walk(root):
#     for name in tqdm(files) :
#         with open (os.path.join(path, name)) as json_file:
#             #print os.path.join(path, name)
#             raw_data = json.load(json_file)
#             try :
#                 unicode_content = raw_data['data']['bloglovin_content']['content']['content']
#                 unicode_content = replace_text(unicode_content)
#                 if len(unicode_content)==0 :
#                     continue
#                 else :#>MINIMUM_OCCUR):
#                     sent= split_sent.tokenize(unicode_content)#.encode('ascii','ignore'))
#                     a_merged = ' '.join(sent)
#                     with open(os.path.join(data_cleaned, name)+'.txt','w') as f:
#                         f.write(a_merged)

#             except :
#                 try:
#                     unicode_content = raw_data['data']['bloglovin_content']['content']
#                     unicode_content = replace_text(unicode_content)
#                     if len(unicode_content)==0 :
#                         continue
#                     else :#>MINIMUM_OCCUR):
#                         sent=split_sent.tokenize((unicode_content).encode('ascii','ignore') )
#                         a_merged = ' '.join(sent)
#                         with open(os.path.join(data_cleaned, name)+'.txt','w') as f:
#                             f.write(a_merged)

#                 except :
#                     try:   
#                         unicode_content = raw_data['data']['mercury_content']['content'] 
#                     except : 
#                         print (name)
#                         print (raw_data)
#                         continue
#                     soup = bs(unicode_content, 'html.parser')
#                     findp = soup.find_all('p') 
#                     gloss_count=0
#                     word_count=0
#                     paralist=[]
#                     for p in findp:
#                         tmp = replace_text(p.text)
#                         if(len(tmp)>1) :
#                             paralist.append(tmp)
            
#                 #print "{},{},{}".format(gloss_ratio,gloss_count,word_count)
#                 # if(gloss_ratio):#>MINIMUM_OCCUR):
#                 #     #print "In ['data']['mercury_content']['content']"
#                     for paras in paralist:    
#                         sent=split_sent.tokenize(paras.encode('ascii','ignore'))
#                         a_merged = ' '.join(sent)
#                         with open(os.path.join(data_cleaned, name)+'.txt','w') as f:
#                             f.write(a_merged)



                


# multikey = {}
# multikey['fabric']= count_terms(recalldict, set(recalldict.keys()) & set(fabric.keys()))
# multikey['fashion_strings']= count_terms(recalldict, set(recalldict.keys()) & set(fashion_strings.keys()))
# multikey['article_types']= count_terms(recalldict, set(recalldict.keys()) & set(article_types.keys()))
# multikey['attribute_types']= count_terms(recalldict, set(recalldict.keys()) & set(attribute_types.keys()))

with open('terms_count_gloss2.txt', 'w') as outfile:
    json.dump(allsent, outfile, sort_keys=True, indent=4, separators=(',', ': '))

# #with open('sents_in_100K.pickle', 'wb') as pklfile:
# #    pickle.dump(allsent, pklfile, protocol=pickle.HIGHEST_PROTOCOL)

# print "fabric = {}".format(len(multikey['fabric']))
# print "article_types = {}".format(len(multikey['article_types']))
# print "fashion_strings = {}".format(len(multikey['fashion_strings']))
# print "attribute_types = {}".format(len(multikey['attribute_types']))

'''
json_keys=[]
for key in json_data :
    json_keys.append(key.encode('ascii','ignore'))

json_keys=list(set(json_keys))
print json_keys
'''
