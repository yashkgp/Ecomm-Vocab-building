import pickle
from collections import defaultdict

#with open("test3010Ann.txt","r") as f :
#    testSent = pickle.load(f)

def get_tags():
    tag_dict = {}
    with open("tags.txt","r") as t :
        i=0
        for line in t :
            tag_dict[line.strip()] = i
            i+=1    
    class_names = ['B-ART','I-ART','B-ATTR','I-ATTR',
               'B-FSN','I-FSN','B-FBR','I-FBR']
    tmp=100
    for cls in class_names :
        if (cls not in tag_dict):
            tag_dict[cls] =tmp
            tmp+=1
    inv_tag_dict = {v:k for k,v in tag_dict.iteritems()}
    return tag_dict, inv_tag_dict


def onlyB(cur,nxt,tag_dict):
    if ( cur == tag_dict['B-ART'] and nxt!=tag_dict['I-ART']) :
        return True, tag_dict['B-ART']
    elif ( cur ==tag_dict['B-ATTR'] and nxt!=tag_dict['I-ATTR']):
        return True, tag_dict['B-ATTR']
    elif ( cur ==tag_dict['B-FBR'] and nxt!=tag_dict['I-FBR']):
        return True, tag_dict['B-FBR']
    elif ( cur ==tag_dict['B-FSN'] and nxt!=tag_dict['I-FSN']):
        return True, tag_dict['B-FSN']
    else :
        return False, -1

## Returns flag, inside tag
def isBITag(cur,nxt,tag_dict):
    if ( cur ==tag_dict['B-ART'] and nxt==tag_dict['I-ART']) :
        return True, tag_dict['B-ART'], tag_dict['I-ART']
    elif ( cur ==tag_dict['B-ATTR'] and nxt==tag_dict['I-ATTR']) :
        return True, tag_dict['B-ATTR'], tag_dict['I-ATTR']
    elif ( cur ==tag_dict['B-FBR'] and nxt==tag_dict['I-FBR']) :
        return True, tag_dict['B-FBR'], tag_dict['I-FBR']
    elif ( cur ==tag_dict['B-FSN'] and nxt==tag_dict['I-FSN']):
        return True, tag_dict['B-FSN'], tag_dict['I-FSN']
    else :
        return False,-1, -1


def endsInB(cur,tag_dict):
    if ( cur ==tag_dict['B-ART'] ):
        return True, tag_dict['B-ART']
    elif ( cur ==tag_dict['B-ATTR'] ) :
        return True, tag_dict['B-ATTR']
    elif ( cur ==tag_dict['B-FBR'] ):
        return True, tag_dict['B-FBR']
    elif ( cur ==tag_dict['B-FSN']):
        return True, tag_dict['B-FSN']
    else :
        return False, -1

###########################
# For test data
##########################


def save_in_Dict(tag_type, word, tag_dict, ART_dict, ATTR_dict, FBR_dict, FSN_dict) :
    word = word.lower()
    if tag_type == tag_dict['B-ART'] :
        ART_dict[word] +=1 #= word
    elif tag_type == tag_dict['B-ATTR'] :
        ATTR_dict[word] +=1 #= word
    elif tag_type == tag_dict['B-FBR'] :
        FBR_dict[word] +=1 #= word    
    elif tag_type == tag_dict['B-FSN'] :
        FSN_dict[word] +=1 #= word
    return ART_dict, ATTR_dict, FBR_dict, FSN_dict


def get_word(y_test, testSent):
    tag_dict, inv_tag_dict = get_tags()
    ART_dict = defaultdict(int)
    ATTR_dict = defaultdict(int)
    FBR_dict = defaultdict(int)
    FSN_dict = defaultdict(int)
    #y_test= pred_lab # prediction data
    numTestData= len(testSent)
    for i, sent in enumerate(testSent):
        sentLength= len(sent)
        for sl in range(sentLength -1) :
            # __B__
            BFlag, tag = onlyB(y_test[i][sl],y_test[i][sl+1],tag_dict)
            if BFlag == True :
                word = testSent[i][sl][0]
                ART_dict, ATTR_dict, FBR_dict, FSN_dict= save_in_Dict(tag, word.strip(),
                 tag_dict, ART_dict, ATTR_dict, FBR_dict, FSN_dict)
                #actualTerms[word.strip()]=word.strip()
            #__BII____
            else:
                BIflag, beg, ins = isBITag(y_test[i][sl],y_test[i][sl+1],tag_dict)
                if BIflag == True :
                    nxt=sl+1
                    word = testSent[i][sl][0]
                    while( (y_test[i][nxt]== ins) and nxt <(sentLength-1)):
                        word= word +' '+ testSent[i][nxt][0]
                        nxt+=1    
                    ART_dict, ATTR_dict, FBR_dict, FSN_dict= save_in_Dict(beg, word.strip(),
                     tag_dict, ART_dict, ATTR_dict, FBR_dict, FSN_dict)
                    #actualTerms[word.strip()]=word.strip()
                    sl=nxt
        #___B
        EBFlag, tag = endsInB(y_test[i][-1],tag_dict)
        if EBFlag == True :
            word = testSent[i][-1][0]
            ART_dict, ATTR_dict, FBR_dict, FSN_dict= save_in_Dict(tag, word.strip(), 
                tag_dict, ART_dict, ATTR_dict, FBR_dict, FSN_dict)
        #actualTerms[word.strip()]=word.strip()   
    return ART_dict, ATTR_dict, FBR_dict, FSN_dict

