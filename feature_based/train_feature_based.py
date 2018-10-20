    
import os
import pickle
import json
import sklearn_crfsuite  
from  sklearn_crfsuite import metrics
import scipy.stats
from sklearn.metrics import make_scorer
#from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from spacy.lang.en import English
parser = English()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

"""Load dataset from a CoNLL format file
"""
def load_dataset(filename):
    with open(filename) as fp:
        tup =()
        sent, doc=[], []
        for line in fp:
            line = line.strip()
            if (len(line) == 0):
                # Empty line and also end of a sentence
                if len(sent) != 0 or line.startswith("-DOCSTART"):
                    doc.append(sent)
                sent =[]
            else :
                if len(line) < 2:
                    continue
                line_content = line.split("\t")
                tup = (line_content[0],line_content[1],line_content[2])
                sent.append(tup)
    print "Completed reading of {} lines of the dataset from file :{} ".format(len(doc),filename )
    return doc

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

def annotate_with_dep_parsing(sent_list,allkey):
    for snt in range(len(sent_list)):
        annSent= sent_list[snt]
        sent=''
        for j in range(len(annSent)):
            #print annSent[j][0]
            sent= sent + ' ' +(annSent[j][0])
        sent= sent.strip()
        parsedSent = parser(unicode(sent,"utf-8"))
        #print snt, parsedSent
        for token in parsedSent :        
            if token.dep_ in ["amod"," compound"] and (token.head.orth_ in allkey) :#or (token.orth_ in allkey)):
                print(token.pos_,token.orth_,  token.dep_, token.head.orth_, token.i)
                sent_list[snt][token.i]+=(1,)
                #print(token.pos_,token.orth_,  token.dep_, token.head.orth_, token.head.i,[t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
        for  j in range(len(annSent)):
            if (len(annSent[j])) ==3 :
                sent_list[snt][j] +=(0,)
    return sent_list

'''
        if token.head.orth_ in allkey :
            print(token.pos_,token.orth_,  token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
            
           #if token.head.pos_ =="VERB" : 
                # ['VERB','NOUN','ADJ'] 
        elif  token.orth_ in allkey:
            print(token.pos_,token.orth_,  token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
           
        if token.dep_=="amod" :
                print token.head.orth_
'''
#######################################
#
# Create features 
#
#######################################
def word2features(sent,i):
    word = sent[i][0]
    postag = sent[i][2]
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

def training_op(X_train,y_train ) :
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=10,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    filename = 'crf_withoutCV'
    pickle.dump(crf, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

    labels = list(crf.classes_)
    labels.remove('O')
    labels
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=5,
                            verbose=1,
                            n_jobs=1,
                            n_iter=5,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)
    crf = rs.best_estimator_
    filename = 'CRF_model'
    pickle.dump(crf, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train_FBR, train_FSN, train_ART, train_ATTR = load_terms("train_terms.json")
    train_keys = dict(train_FBR.items() + train_FSN.items() + train_ART.items() + train_ATTR.items())
    train_sents  = load_dataset("train_2704_zalando.txt")
    #train_sents = annotate_with_dep_parsing(train_sents,train_keys)
    check = {}
    actualTokens=[sent2tokens(s) for s in train_sents]
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    # check['actualTokens'] =actualTokens
    # check['X_train'] = X_train
    # check['y_train'] = y_train
    # with open('check.json','w') as outfile :
    #     json.dump(check, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    training_op(X_train,y_train )



