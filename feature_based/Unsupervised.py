#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 07:19:07 2017

@author: mi0240
"""

#import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')
import codecs
import os
import json
from bs4 import BeautifulSoup as bs
from collections import defaultdict,Counter
import pickle
import math
import csv
import unicodedata
import re
import nltk.data
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
FREQ_THRESHOLD = 5
MULTI_WORD = True
"""Load dataset from a CoNLL format file
"""

def load_sentences(filename):
    #with codecs.open(filename, "r",encoding='utf-8', errors='ignore') as fp:
    with open(filename) as fp:
        doc_words, doc_tags, doc_pos = [], [], []
        sent_words, sent_tags, sent_pos = "", "", ""
        for line in fp:
            line = line.strip()
            if (len(line) == 0):
                # Empty line and also end of a sentence
                if len(sent_words) != 0 or line.startswith("-DOCSTART"):
                    sent_words=sent_words.encode('ascii','ignore')
                    doc_words.append(sent_words.strip())
                sent_words, sent_tags, sent_pos = "", "", ""
            else :
                if len(line) < 2:
                    continue
                line_content = line.split("\t")
                sent_words= sent_words + str(line_content[0])+" "
    print "Completed reading of {} lines of the dataset from file :{} ".format(len(doc_words),filename )
    #return dataset_words, dataset_tags, dataset_pos
    return doc_words#, doc_tags, doc_pos


#######################################
#
# Filter for NP and also find context word
#
#######################################

def NP_and_contextword(allsent,grammar) :
    lemmatizer = WordNetLemmatizer()
    split_sent = nltk.data.load('tokenizers/punkt/english.pickle')
    stop_words = set(stopwords.words('english'))
    contextDict = defaultdict(list)
    extract = []
    for lines in allsent :
        #l = lines.encode('ascii','ignore')
        word_tokens = word_tokenize(lines)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        pos = (pos_tag(filtered_sentence))
        for g in grammar:
            cp = nltk.RegexpParser(g)
            #pos = (pos_tag(word_tokenize(filtered_sentence))) 
            #print " ".join(["{}/{}".format(word,postag) for word, postag in pos])
            result = cp.parse(pos) 
            #print result.flatten()
            #for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
            for subtree in result.subtrees():
                flag =0 
                if subtree.label()=='NP':
                    # print the noun phrase as a list of part-of-speech tagged words
                    tree_leaves =  subtree.leaves()
                    if len(tree_leaves) <10 :
                        flag=1
                        #extract.append( " ".join(["{}".format(lemmatizer.lemmatize(word))
                        #for word, postag in tree_leaves ]))
                        cwpos=[]
                        getcw = ''
                        for word, postag in subtree.leaves():#tree_leaves :
                            cwpos.append((word, postag))
                            #lemmatizer.lemmatize(word)
                            getcw = getcw+" "+(word)
                            #print getcw
                        extract.append(getcw.strip())
                '''
                if flag==0 :
                     allsent.remove(lines)
                '''
                #######################################
                ### FInd context
                #######################################
                if flag==1 :
                    cwposlen=len(cwpos)
                    poslen=len(pos)
                    for i in range(poslen):
                        if pos[i]==cwpos[0] and pos[i+cwposlen-1]==cwpos[-1] :
                            start_ind = i-1
                            end_ind = i+cwposlen
                            break
                    for i in range(3) :
                        if start_ind-i > 0 :
                            if pos[start_ind-i][1] in ['NN','VB','JJ']:
                                #print pos[start_ind-i]
                                contextDict[extract[-1]].append(pos[start_ind-i][0])
                        if end_ind+i < poslen :
                            if pos[end_ind+i][1] in ['NN','VB','JJ']:
                                #print pos[end_ind+i]   
                                contextDict[extract[-1]].append(pos[end_ind+i][0])
                                #['VB', 'VBD', 'VBG', 'VBN','VBP','VBZ' ]:
    print "done extraction"
    return extract, contextDict

       
#######################################
#
# Frequency and length calculation
#
#######################################




## Filter the extracted terms having frequency >= threshold
def filter_freq_threshold(extracted_NP, threshold):  
    counts=[]
    terms=[]  
    freq= []
    freq_list = Counter(extracted_NP)
    for k,v in freq_list.iteritems() :
        if v > threshold :
            terms.append(k)
            counts.append(v)
            freq.append([k,v,len(k.split())])
    return freq, terms, counts


#######################################
#
# Remove terms containing stoplist
#
#######################################


def remove_frequent_terms(candidateTerm, contextDict):
    stoplist =['/','%','@','very','new','good','bad','next','every','last','average',
               'different','young', 'enough','few','first','great','.','=',
               u'\u2019',u'\u201c',u'\u201d' ]
    for f in candidateTerm :
        for sl in stoplist :
            if sl in f[0]: 
                print f[0]
                candidateTerm.remove(f)
                del worddict[f[0]]
                break
    for k in contextDict.keys():
        for sl in stoplist :
            if sl in k:
                del contextDict[k]
                break
    return candidateTerm, contextDict


#######################################
#
# C-value calculation
#
#######################################
def calculate_Cval(candidateTerm, worddict, cval_threshold):
    for i in range(len(candidateTerm)):
        temp={}
        sumval=0
        cand_str = candidateTerm[i][0]
        cand_freq = worddict[candidateTerm[i][0]] #candidateTerm[i][1]
        cand_len = candidateTerm[i][2]
        
        for j in range (0, i):
            longstring = candidateTerm[j][0]
            if cand_str in longstring :
                if candidateTerm[j][2]==cand_len+1:
                    worddict[cand_str]= cand_freq-candidateTerm[j][1] 
                temp[longstring] = worddict[longstring]
                sumval= sumval+worddict[longstring]
        #print temp , sumval, cand_str  
        if len(temp)==0 :
            #print cand_len,cand_freq
            getval = math.log(cand_len,2)*cand_freq
            candidateTerm[i].append(getval)
            
        else :
            num_long_term=len(temp)
            getval = math.log(cand_len,2)*(cand_freq-(sumval/num_long_term))
            candidateTerm[i].append(getval)
    #cndtrm=list(copy.deepcopy(candidateTerm))  
    c_val=[]
    for i in range(len(candidateTerm)):
        if candidateTerm[i][3] >= cval_threshold :
            c_val.append(candidateTerm[i])
    with open("c_val_list.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(c_val)
    return c_val#,candidateTerm


    
    
#######################################
#
# weight and N-Value calculation
#
#######################################

def sortCval(item):
    return item[3]

def calculate_Nval(c_val,contextDict):  
    c_val.sort(key = sortCval, reverse = True)         
    num_c=int(len(c_val) * 0.2)
    contextlist = Counter()
    for i in range(num_c):
        key=c_val[i][0]
        vals = list(contextDict[key])
        for cwd in vals:
            contextlist[cwd]+=1
    slist=['/','.','*','@','-year','+','%']
    for k in list(contextlist):
            if contextlist[k] < 2:
                del contextlist[k]
            else :
                if any(ext in k for ext in slist):
                    del contextlist[k]
    for i in range(len(c_val)):
        candstr=c_val[i][0]
        context_count= Counter(contextDict[candstr])
        NC=0
        for word in context_count.keys():
            if word in contextlist.keys():
                # num_c is for the calculation of weight
                NC += context_count[word] * float(contextlist[word])/num_c 
        c_val[i].append(NC)
    return c_val


'''
    saveC=[]    
    for i in c_val:
        saveC.append(i)
'''

    

#######################################
#
# NC value calculation
#
#######################################

def sortNCval(item):
    return item[5]

def calculate_NCval(c_val, alpha, beta):
    for i in  range(len(c_val)):
        c_val[i].append(alpha*c_val[i][3] + beta*c_val[i][4])
    c_val.sort(key = sortNCval, reverse = True)
    return c_val


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


train_sents  = load_sentences("train_2704.txt")
test_sents  = load_sentences("test_2704.txt")
#tmp = train_sents + test_sents
allsent =  list(set(train_sents + test_sents))
#del tmp
if (MULTI_WORD):
    grammar = [ "NP: { <NNP | NNPS>* <JJ| NN| NNS>+ <NN | NNS>+}"]
else :
    grammar = [ "NP: { <JJ>* < NN| NNS| NNP>+}", "NP:{< NN| NNS>}"]
extract, contextDict = NP_and_contextword(allsent,grammar) 
filename = 'extract'
pickle.dump(extract, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
filename = 'contextDict'
pickle.dump(contextDict, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

candidateTerm, terms, counts = filter_freq_threshold (extract, FREQ_THRESHOLD)
worddict = dict(zip(terms,counts))
candidateTerm, contextDict = remove_frequent_terms(candidateTerm, contextDict)


filename = 'worddict'
pickle.dump(worddict, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
filename = 'candidateTerm'
pickle.dump(candidateTerm, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
filename = 'contextDict'
pickle.dump(contextDict, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

if (MULTI_WORD):
    cval_threshold =10
else 
    cval_threshold = 0
c_val = calculate_Cval(candidateTerm, worddict, cval_threshold)

filename = 'c_val'
pickle.dump(c_val, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)


c_val = calculate_Nval(c_val,contextDict) #Nval appended beside cval
alpha = 0.7
beta = 0.3
c_val = calculate_NCval(c_val, alpha, beta)

filename = 'c_nc_val'
pickle.dump(c_val, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

#######################################
#
#
#
#######################################
import json

train_FBR, train_FSN, train_ART, train_ATTR = load_terms("train_terms.json")
test_FBR, test_FSN, test_ART, test_ATTR = load_terms("test_terms.json")
train_keys = dict(train_FBR.items() + train_FSN.items() + train_ART.items() + train_ATTR.items())
test_keys = dict(test_FBR.items() + test_FSN.items() + test_ART.items() + test_ATTR.items())

allkey = dict(train_keys.items()+ test_keys.items())
glossary_terms=[]
def find_order (c_valList, whole, part, partFull):
    print len
    wfp = open(whole, "w")
    pfp = open(part, "w")
    wpfp = open(partFull, "w")
    matching=[0] * len(c_valList)
    found =0
    found_split = 0
    j=0
    for i in c_valList:
        matching[j]=0
        if i[0] in allkey :
            #print 'whole match : '+i[0]
            wfp.write("%s\n" % i[0])
            found += 1
            matching[j]=1
            glossary_terms.append([i[0],i[3]])
        else :
            split_cand = i[0].split()
            count = 0
            for sw in  range(len(split_cand)):
                if split_cand[sw] in allkey :
                    count += 1
                    pfp.write("%s " % split_cand[sw])
            if count > 0:#== split_cand :
                found_split += 1
                if count==len(split_cand):
                    wpfp.write("%s \n" % i[0])
                else :
                    pfp.write(" :: %s \n" % i[0])
                    #matching[j]=1
                    glossary_terms.append([i[0],i[3]])
        j+=1
    wfp.close()
    pfp.close()
    wpfp.close()
    return matching

#c_val.sort(key = getKey)
c_val.sort(key = sortCval,reverse=True)
c_match = find_order (c_val,
                      "c_val_whole.txt",
                      "c_val_part.txt",
                      "c_val_partfull.txt")
c_val.sort(key = sortNCval,reverse=True)
nc_match = find_order (c_val,
                       "nc_val_whole.txt",
                      "nc_val_parttxt",
                      "nc_val_partfull.txt")

print c_match == nc_match
print Counter(c_match)
print Counter(nc_match)

def writeCSV(listname,filename): 
    import csv
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(listname)

#writeCSV(saveC,"nc_val_list.csv",)

'''
#######################################
#
# Find MAP c: 0.4213346157320142
#          nc:0.4231938862878412
#######################################
def mAPmAR( matchedlist, relevant):
    ar=[]
    ap= []
    for i in range(len(matchedlist)):
        if matchedlist[i]==1 :
            c= Counter(matchedlist[0:i])
            ap.append( float(c[1])/i)
            ar.append( float(c[1])/relevant)
    mAP= float(sum(ap))/relevant       
    mAR= float(sum(ar))/relevant
    import matplotlib.pyplot as plt
    import numpy as np
    # Create a trace
    x = np.asarray(ar)
    y = np.asarray(ap)
    
    plt.plot(x, y)
    return mAP #, mAR

relevant=Counter(c_match)[1]
print mAPmAR(c_match,relevant)
print mAPmAR(nc_match,relevant)
'''

#######################################
#
#
#
#######################################
