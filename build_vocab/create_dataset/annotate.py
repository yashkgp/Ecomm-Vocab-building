
import pickle
import json
from nltk import pos_tag, word_tokenize
import pandas as pd
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

def load_pkl_file(infile, mode):
    with open(infile, mode) as pklfile:
        sent=pickle.load(pklfile)
        return sent

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

def is_not_tagged(tags,start,end):
    count=1
    for pl in range(start+1,end):
        if(tags[pl]=='O') :
            count +=1
    if count==len(tags):
        return 1
    else :return 0

def BIO(l, words,tags,dicts,BVal,IVal) :
    for k,v in dicts.iteritems():
        splitKey=k.split()
        keyLength = len(splitKey)
        #print k ,splitKey
        try :
            if keyLength> 2 and k in l.lower():
                #print ("splitKey: "+splitKey)
                word = " ".join(words).lower()
                #print ("word: "+ word)
                start = words.index(splitKey[0])
                end = start+keyLength
                if(is_not_tagged):
                    tags[start] = BVal
                    for pl in range(start+1,end):
                        tags[pl] = IVal
            elif keyLength==2 and k in l.lower() :
                word = " ".join(words).lower()
                ind0 = words.index(splitKey[0])
                ind1 = ind0+1
                #print ind0, '-',ind1
                if tags[ind0]=='O' and tags[ind1]=='O'and words[ind1].find(splitKey[1])!=-1:
                    tags[ind0] = BVal
                    tags[ind1] = IVal
    
            elif keyLength==1 and k in words:
                 ind0 = words.index(k)
                 if(tags[ind0]=='O'):
                    tags[ind0] = BVal
        except :
            print 'skipping ::' +l 
    return tags

  
def annotateSent(allsent, FBR ,FSN, ART, ATTR) :
    all_sent=[]
    all_pos=[]
    all_tag=[]
    for i in allsent:
        l = i
        #print l
        wordToken = word_tokenize(l)
        posToken = pos_tag(wordToken)
        pos=[]
        for ind,ps in enumerate(posToken):
            pos.append(ps[1])
        tags = ['O'] * len(pos)
        #words = [word for word in wordToken]
        #BIOBrand(l,wordToken,pos,brands,'B-BRD','I-BRD') 
        words = [word.lower() for word in wordToken]
        tags = BIO(l,words,tags,FBR,'B-FBR','I-FBR') 
        tags = BIO(l,words,tags,FSN,'B-FSN','I-FSN')
        tags = BIO(l,words,tags,ART,'B-ART','I-ART') 
        tags = BIO(l,words,tags,ATTR,'B-ATTR','I-ATTR') 
        all_sent.append(wordToken)
        all_pos.append(pos)
        all_tag.append(tags)
    return all_sent, all_pos, all_tag


def writeBIO(outfile,sent_list,tag,pos) :
    sent_all = []
    for st in range(len(sent_list)) :
        sent = [ ] 
        for word in range(len(sent_list[st])) :
            sent =[]
            sent.append(st+word)
            sent.append(st)
            sent.append(sent_list[st][word])
            sent.append(pos[st][word])
            sent.append(tag[st][word])
            sent_all.append(sent)
    print(len(sent_all))
    print (len(sent_all[0]))
    sent_all = np.array(sent_all)
    print (sent_all.shape)
    df_sent = pd.DataFrame(sent_all,columns = ['ID','Sentence #','Word','POS','Tag'])
    df_sent.to_csv(outfile,sep=',',encoding='utf-8')
    # outf= open(outfile,"w")
    # for st in range(len(sent_list)) :
    #     for word in range(len(sent_list[st])) : 
    #         outf.write(sent_list[st][word].encode('utf-8').strip()+"\t"
    #          +tag[st][word].encode('utf-8').strip()+"\t"
    #          +pos[st][word].encode('utf-8').strip()+"\n")
    #     outf.write("\n")
    # outf.close()

#######################################
#
# Train and test data
#
#######################################   

#703262, 2168410
#849164, 321777
if __name__ == '__main__':
    train_sent = load_pkl_file ('train_100K_riverisland.pkl','rb')
    test_sent = load_pkl_file ('test_100K_riverisland.pkl','rb')
    #fabric, fashion_strings, article_types, attribute_types= load_terms("terms_count_gloss_100K.json")
    train_FBR, train_FSN, train_ART, train_ATTR = load_terms("train_terms_new2.json")
    test_FBR, test_FSN, test_ART, test_ATTR = load_terms("test_terms_new2.json")

    FBR = dict(train_FBR.items() + test_FBR.items())
    FSN = dict(train_FSN.items() + test_FSN.items())
    ART = dict(train_ART.items() + test_ART.items())
    ATTR = dict(train_ATTR.items() + test_ATTR.items())

    train_data, train_pos, train_tag = annotateSent(train_sent,FBR ,FSN, ART, ATTR)  
    test_data, test_pos, test_tag = annotateSent(test_sent,FBR ,FSN, ART, ATTR)
    total_data = np.concatenate((np.array(train_data),np.array(test_data)),axis = 0)
    total_pos = np.concatenate((np.array(train_pos),np.array(test_pos)),axis = 0)
    total_tag  = np.concatenate((np.array(train_tag),np.array(test_tag)),axis = 0)
 
    writeBIO("train_2704_riverisland.csv",train_data, train_tag, train_pos )
    writeBIO("test_2704_riverisland.csv",test_data, test_tag, test_pos)

