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

from train_feature_based import load_dataset, fashionTerms, load_terms
from train_feature_based import annotate_with_dep_parsing, word2features, sent2features, sent2labels, sent2tokens
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
if __name__ == '__main__':
    train_FBR, train_FSN, train_ART, train_ATTR = load_terms("train_terms.json")
    train_keys = dict(train_FBR.items() + train_FSN.items() + train_ART.items() + train_ATTR.items())
    train_sents  = load_dataset("train_2704.txt")
    trainTokens=[sent2tokens(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    class_names = ['B-ART','I-ART','B-ATTR','I-ATTR',
                   'B-FSN','I-FSN','B-FBR','I-FBR']
    for i in range(len(class_names)/2):
        print class_names[i*2],class_names[2*i+1]
        B = class_names[i*2]
        I = class_names[2*i+1]
       
        trainTerms = extractedTerms(y_train,trainTokens,B,I)
        # Test_terms[B] = actualTerms
        # Predicted_Terms[B] = predictedTerms
        # actualTerms=set([x.lower() for x in actualTerms])
        # predictedTerms=set([x.lower() for x in predictedTerms])
        trainTerms=set([x.lower() for x in trainTerms])
        # predicted_wrong = list(set(actualTerms) - set(actualTerms)&set(predictedTerms))
        # pred_wrong ={}
        # for i in predicted_wrong:
        #     pred_wrong[i]=i
        # predicted_wrong_all[B] = pred_wrong
        # new_terms_in_test=set(actualTerms) - set(trainTerms)
        # print 'Terms it testData = '+str(len(actualTerms))
        # print 'Terms it prediction = '+str(len(predictedTerms))
        print 'Terms it training = '+str(len(trainTerms))
        #print 'Common in test and train = '+str(len(set(actualTerms)&set(trainTerms)))
        # print 'new in test =' +str(len(new_terms_in_test))
        # print 'new term correctly predicted' + str(len(set(predictedTerms)&set(actualTerms)))