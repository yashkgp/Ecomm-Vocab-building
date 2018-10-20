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

from train_feature_based import  fashionTerms, load_terms
from train_feature_based import annotate_with_dep_parsing

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


def load_pkl_file(infile):
    with open(infile) as pklfile:
        sent = [line.decode('utf-8').strip() for line in pklfile.readlines()]
    sents = [replace_text(line) for line in sent]
    return sents


def extractedTerms(y_test,actualTokens,beg,ins):
    numTestData= len(actualTokens)
    actualTerms={}
    for i in range(numTestData):
        sentLength= len(actualTokens[i])
        for sl in range(sentLength -1) :
            
            if y_test[i][sl]==beg:
                word = actualTokens[i][sl]+" "+actualTokens[i][sl+1]
                actualTerms[word.strip()]=word.strip()
            # elif y_test[i][sl].startswith(beg) and y_test[i][sl+1].startswith(ins) and y_test[i][sl][2:]==y_test[i][sl+1][2:]:
            #     nxt=sl+1
            #     word = actualTokens[i][sl]
            #     while(y_test[i][nxt].startswith(ins) and nxt <(sentLength-1)):
            #         word= word +' '+ actualTokens[i][nxt]
            #         nxt+=1    
            #     actualTerms[word.strip()]=word.strip()
            #     sl=nxt

        if y_test[i][-1]==beg:
            word = actualTokens[i][-1]
            actualTerms[word.strip()]=word.strip()
    return actualTerms

def evaluate(test_sents, modelfile, mode):
    # load the model
    X_test = [sent2features(s) for s in test_sents]
    loaded_model = pickle.load(open(modelfile, mode))
    labels = list(loaded_model.classes_)
    labels.remove('O')
    labels
    y_pred = loaded_model.predict(X_test)
   

    return  y_pred
def load_dataset(filename):
    with open(filename) as fp:
        tup =()
        sent, doc=[], []
        for line in fp:
            line = line.decode('utf-8').strip()
            if (len(line) == 0):
                # Empty line and also end of a sentence
                if len(sent) != 0 or line.startswith("-DOCSTART"):
                    doc.append(sent)
                sent =[]
            else :
                if len(line) < 2:
                    continue
                line_content = line.split("\t")
                tup = (line_content[0],line_content[1])
                sent.append(tup)
    print "Completed reading of {} lines of the dataset from file :{} ".format(len(doc),filename )
    return doc


def annotateSent(allsent) :
    all_sent=[]
    all_pos=[]
    for i in allsent:
        l = i
        #print l
        wordToken = word_tokenize(l)
        posToken = pos_tag(wordToken)
        pos=[]
        for ind,ps in enumerate(posToken):
            pos.append(ps[1])
        #words = [word for word in wordToken]
        #BIOBrand(l,wordToken,pos,brands,'B-BRD','I-BRD') 
        words = [word.lower() for word in wordToken] 
        all_sent.append(wordToken)
        all_pos.append(pos)
    return all_sent, all_pos

def writeBIO(outfile,sent_list,pos) :
    # sent_all = []
    # for st in range(len(sent_list)) :
    #     sent = [ ] 
    #     for word in range(len(sent_list[st])) :
    #         sent =[]
    #         sent.append(st+word)
    #         sent.append(st)
    #         sent.append(sent_list[st][word])
    #         sent.append(pos[st][word])
    #         sent.append(tag[st][word])
    #         sent_all.append(sent)
    # print(len(sent_all))
    # print (len(sent_all[0]))
    # sent_all = np.array(sent_all)
    # print (sent_all.shape)
    # df_sent = pd.DataFrame(sent_all,columns = ['ID','Sent','Word','POS','Tag'])
    # df_sent.to_csv(outfile,sep=',',encoding='utf-8')
    outf= open(outfile,"w")
    for st in range(len(sent_list)) :
        for word in range(len(sent_list[st])) : 
            outf.write(sent_list[st][word].encode('utf-8').strip()+"\t"
             +pos[st][word].encode('utf-8').strip()+"\n")
        outf.write("\n")
    outf.close()



def writeBIO2(outfile,sent_list,tag,pos) :
    # sent_all = []
    # for st in range(len(sent_list)) :
    #     sent = [ ] 
    #     for word in range(len(sent_list[st])) :
    #         sent =[]
    #         sent.append(st+word)
    #         sent.append(st)
    #         sent.append(sent_list[st][word])
    #         sent.append(pos[st][word])
    #         sent.append(tag[st][word])
    #         sent_all.append(sent)
    # print(len(sent_all))
    # print (len(sent_all[0]))
    # sent_all = np.array(sent_all)
    # print (sent_all.shape)
    # df_sent = pd.DataFrame(sent_all,columns = ['ID','Sent','Word','POS','Tag'])
    # df_sent.to_csv(outfile,sep=',',encoding='utf-8')
    outf= open(outfile,"w")
    for st in range(len(sent_list)) :
        for word in range(len(sent_list[st])) : 
            outf.write(sent_list[st][word].encode('utf-8').strip()+"\t"
             +tag[st][word].encode('utf-8').strip()+"\t"
             +pos[st][word].encode('utf-8').strip()+"\n")
        outf.write("\n")
    outf.close()
def word2features(sent,i):
    word = sent[i][0]
    postag = sent[i][1]
    #amodcomp= sent[i][3]
    #BIOtag = sent[i][2]
    # Common features for all words
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2], 
        #'AmodCompound': amodcomp,
        'lemme':lemmatizer.lemmatize(word),
        #'BIOtag': BIOtag,
    }
     # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        #BIOtag1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            #'-1:BIOtag': BIOtag1,
        })
    else:
        features['BOS'] = True
    
    # Features for words that are not at the end of a document
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        #BIOtag1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
           # '+1:BIOtag': BIOtag1,
        })
    else:
        features['EOS'] = True            
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [i[1] for i in sent]

def sent2tokens(sent):
    return [i[0] for i in sent]
if __name__ == '__main__':
    test_sent = load_pkl_file ('zalando.csv')  
    test_data, test_pos = annotateSent(test_sent)
    writeBIO("test_2704new2.txt",test_data, test_pos)
    test_sents  = load_dataset("test_2704new2.txt")
    y_pred = evaluate(test_sents, "CRF_model", "rb")
    class_names = ['B-ART','I-ART','B-ATTR','I-ATTR',
					'B-FSN','I-FSN','B-FBR','I-FBR']
    extract_pred ={}
    actualTokens=[sent2tokens(s) for s in test_sents]
    for i in range(len(class_names)/2):
        print class_names[i*2],class_names[2*i+1]
        B = class_names[i*2]
        I = class_names[2*i+1]
  
        predictedTerms = extractedTerms(y_pred,actualTokens,B,I)
        
        predictedTerms=set([x.lower() for x in predictedTerms])
        pred ={}
    
        for i in predictedTerms :
            pred[i]=i
        extract_pred[B]=pred
    with open('extract_pred_zalando.json','w') as outfile :
        json.dump(extract_pred, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    writeBIO2("pred_netaporter.txt",test_data,y_pred,test_pos)



