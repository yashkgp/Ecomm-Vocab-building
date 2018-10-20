import pickle 
from collections import defaultdict

def onlyB(cur,nxt):
    if ( cur == 'B-ART' and nxt!= 'I-ART') :
        return True, 'B-ART'
    elif ( cur =='B-ATTR' and nxt!='I-ATTR'):
        return True, 'B-ATTR'
    elif ( cur =='B-FBR' and nxt!='I-FBR'):
        return True, 'B-FBR'
    elif ( cur =='B-FSN' and nxt!='I-FSN'):
        return True, 'B-FSN'
    else :
        return False, -1

## Returns flag, inside tag
def isBITag(cur,nxt):
    if ( cur =='B-ART' and nxt=='I-ART') :
        return True, 'B-ART', 'I-ART'
    elif ( cur =='B-ATTR' and nxt=='I-ATTR') :
        return True, 'B-ATTR', 'I-ATTR'
    elif ( cur =='B-FBR' and nxt=='I-FBR') :
        return True, 'B-FBR', 'I-FBR'
    elif ( cur =='B-FSN' and nxt=='I-FSN'):
        return True, 'B-FSN', 'I-FSN'
    else :
        return False,-1, -1


def endsInB(cur):
    if ( cur =='B-ART'):
        return True, 'B-ART'
    elif ( cur =='B-ATTR' ) :
        return True, 'B-ATTR'
    elif ( cur =='B-FBR' ):
        return True, 'B-FBR'
    elif ( cur =='B-FSN'):
        return True, 'B-FSN'
    else :
        return False, -1



def save_in_Dict(tag_type, word, ART_dict, ATTR_dict, FBR_dict, FSN_dict) :
    word =word.lower()
    if tag_type == 'B-ART' :
        ART_dict[word] +=1 #= word
    elif tag_type == 'B-ATTR' :
        ATTR_dict[word] +=1 # = word
    elif tag_type == 'B-FBR' :
        FBR_dict[word] +=1 # = word        
    elif tag_type == 'B-FSN' :
        FSN_dict[word] +=1 # = word
    return ART_dict, ATTR_dict, FBR_dict, FSN_dict

def get_terms(trainSent) :
    ART_dict = defaultdict(int)
    ATTR_dict = defaultdict(int)
    FBR_dict = defaultdict(int)
    FSN_dict = defaultdict(int)
    for i, sent in enumerate(trainSent):
        sentLength= len(sent)
        for sl in range(sentLength -1) :
        #while sl< (sentLength -1) :
            # __B__
            BFlag, tag = onlyB(trainSent[i][sl][1],trainSent[i][sl+1][1])
            if BFlag == True :
                word = trainSent[i][sl][0]
                #print word
                ART_dict, ATTR_dict, FBR_dict, FSN_dict = save_in_Dict(tag, word.strip(),
                    ART_dict, ATTR_dict, FBR_dict, FSN_dict)
                #actualTerms[word.strip()]=word.strip()
            #__BII____
            else:
                BIflag, beg, ins = isBITag(trainSent[i][sl][1],trainSent[i][sl+1][1])
                if BIflag == True :
                    nxt=sl+1
                    word = trainSent[i][sl][0]
                    while( (trainSent[i][nxt][1]== ins) and nxt <(sentLength-1)):
                        word= word +' '+ trainSent[i][nxt][0]
                        nxt+=1    
                    #print word
                    ART_dict, ATTR_dict, FBR_dict, FSN_dict = save_in_Dict(beg, word.strip(),
                        ART_dict, ATTR_dict, FBR_dict, FSN_dict)
                    #actualTerms[word.strip()]=word.strip()
                    sl=nxt
        #___B
        EBFlag, tag = endsInB(trainSent[i][-1][1])
        if EBFlag == True :
            word = trainSent[i][-1][0]
            ART_dict, ATTR_dict, FBR_dict, FSN_dict = save_in_Dict(tag, word.strip(),
                ART_dict, ATTR_dict, FBR_dict, FSN_dict)
            #actualTerms[word.strip()]=word.strip()  
    return ART_dict, ATTR_dict, FBR_dict, FSN_dict

