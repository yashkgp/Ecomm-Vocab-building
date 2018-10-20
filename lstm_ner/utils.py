import numpy as np
import pickle
from random import shuffle
from config import Config
from build_data import load_vocab
# Special Tokens
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "$NONE$"

import sys  


def load_dataset(filename):
    """Load dataset from a CoNLL format file

    Args:
        filename: name of the dataset file

    Returns:
        a list of list of words, tags, pos.
        doc_words : list of list of words
                         [Sentences  [words]    ]  
        doc_tags : same shape as doc_words, contains
                       corresponding tags
    """
    reload(sys)  
    sys.setdefaultencoding('utf8')
    with open(filename) as fp:
        doc_words, doc_tags, doc_pos = [], [], []
        sent_words, sent_tags, sent_pos = [], [], []
        for line in fp:
            line = line.strip()
            if (len(line) == 0):
                # Empty line and also end of a sentence
                if len(sent_words) != 0 or line.startswith("-DOCSTART"):
                    doc_words.append(sent_words)
                    doc_tags.append(sent_tags)
                    doc_pos.append(sent_pos)
                sent_words, sent_tags, sent_pos = [], [], []
            else :
                if len(line) < 2:
                    continue
                line_content = line.split("\t")
                sent_words.append(line_content[0].encode('ascii','ignore'))
                #print line_content[0]
                sent_tags.append(line_content[1])
                sent_pos.append(line_content[2])
    print "Completed reading of {} lines of the dataset from file :{} ".format(len(doc_words),filename )
  
    #return dataset_words, dataset_tags, dataset_pos
    return doc_words, doc_tags, doc_pos

def word_to_char(word, vocab_chars):
    char_ids = []
    default_char = "#"
    for c in word:
        if c in vocab_chars: 
            char_ids+=[vocab_chars[c]]
            #char_ids.append(vocab_chars.get( c, vocab_chars[default_char]))
    return char_ids

def get_minibatch(data, batch_size):
    
    x_batch, y_batch = [], []
    x_train = data[0]
    y_train = data[1]
    for i,_ in enumerate(x_train):
        if len(x_batch) == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x_train[i]]
        y_batch += [y_train[i]]
    if len(x_batch) != 0:
        yield x_batch, y_batch


def pad_char_ids(config, sentences, pad_tok = 0):

    """
    Args:
        sentences: list of list of list
        padding_value: char value to pad (scalar) 

    Returns:
        padded_sentences: list of list of list
                        (no. of sentences x max_sentence_len x max_word_length)
        word_lengths: list of list
                      (no. of sentences x max_sentence_len)                   
    """


    lengths = [len(sent) for sent in sentences]
    max_sentence_len = max(lengths)

    lengths = [len(w[0]) for sent in sentences for w in sent]
    max_word_length = max(lengths)
    #print max_sentence_len
    #print max_word_length
    padding_value_list = [pad_tok] * max_word_length
    
    word_lengths = []
    padded_sentences = []

    for sent in sentences:
        char_id_words_list = [word[0] for word in sent]
        new_char_id_words_list = [] # Will contain list of list of char_ids (for all words in a sentence)
        word_lengths_sent = []
        if len(sent) < max_sentence_len:
            for char_ids in char_id_words_list:
                if len(char_ids) < max_word_length:
                    char_padding = [pad_tok] * (max_word_length - len(char_ids))
                    new_char_ids = char_ids + char_padding
                    new_char_id_words_list.append(new_char_ids)
                    word_lengths_sent.append(len(char_ids))
                else:
                    new_char_id_words_list.append(char_ids)
                    word_lengths_sent.append(len(char_ids))

            new_char_id_words_list = new_char_id_words_list + [padding_value_list] * (max_sentence_len - len(sent))

            word_lengths_sent = word_lengths_sent + [0] * (max_sentence_len - len(sent))

            padded_sentences.append(new_char_id_words_list)
            word_lengths.append(word_lengths_sent)

        else:
            for char_ids in char_id_words_list:
                if len(char_ids) < max_word_length:
                    char_padding = [pad_tok] * (max_word_length - len(char_ids))
                    new_char_ids = char_ids + char_padding
                    new_char_id_words_list.append(new_char_ids)
                    word_lengths_sent.append(len(char_ids))
                else:
                    new_char_id_words_list.append(char_ids)
                    word_lengths_sent.append(len(char_ids))
            padded_sentences.append(new_char_id_words_list)
            word_lengths.append(word_lengths_sent)

    return padded_sentences, word_lengths


def pad_word_pos_ids(config, sentences, pad_tok =0, get_pos=0 ) :
    length = [len(sent) for sent in sentences]
    max_sent_len = max(length)
    padded_sent=[]
    #padded_pos=[]
    if config.use_chars:
        if not get_pos:
            id_index =1 #word_id_index 
        else :
            id_index =2  #pos_id_index
    else:
        if not get_pos:
            id_index= 0 #word_id_index
        else :
            id_index = 1 #pos_id_index 
         
    for sent in sentences:
        word_ids = [word[id_index] for word in sent]
        #pos_ids = [word[pos_id_index] for word in sent]
        if len(sent) < max_sent_len:
            padding = [pad_tok] * (max_sent_len - len(sent))
            new_sent = word_ids + padding

            padded_sent.append(new_sent)
            #padded_pos.append(pos_ids+padding)
        else:
            padded_sent.append(word_ids)
            #padded_pos.append(pos_ids)

    return padded_sent, length 


def pad_labels(config, labels, pad_tok =0) :
    length = [len(lab) for lab in labels]
    max_sent_len = max(length)
    padded_label=[]

    for lab in labels:
        labels_ids = lab
        if len(lab) < max_sent_len:
            padding = [pad_tok] * (max_sent_len - len(labels_ids))
            new_sent = labels_ids + padding
            padded_label.append(new_sent)
        else:
            padded_label.append(labels_ids)
    return padded_label
   

def vectorize_data(data, vocab_words, vocab_tags, vocab_pos, 
    vocab_chars, config, lowercase =False):
    """ Vectorize text data, also replaces numbers with NUM token 
                            and converts all words to lowercase
    Args:
        data: tuple of words, tags, pos (each as list of list)
        others: dictionary for words, tags, pos

    Returns:
            Vectorized dataset: is a tuple (words, tags, pos)
            which is a list of sentences, each sentence is a list of words
            each word is a tuple (list_of_char_ids, word_id)
    """

    dataset_words = data[0]
    dataset_tags = data[1]
    dataset_pos = data[2]
    #print dataset_words
    #print len(dataset_words)
    #print len(dataset_words[0])
    vec_words = []
    vec_tags = []
    for sentind,_ in enumerate(dataset_words):
        sent_words = dataset_words[sentind]
        sent_tags = dataset_tags[sentind]
        sent_pos = dataset_pos[sentind]
        #print sent_words
        cwp = [] #char, word, pos
        t = [] #tag
        for wordind,_ in enumerate(sent_words):
            word = sent_words[wordind]
            tag = sent_tags[wordind]
            pos = sent_pos[wordind]
            
            if config.use_chars:
                if word.isdigit():
                    t.append(vocab_tags[tag])
                    cwp.append(( word_to_char(word,vocab_chars), 
                        vocab_words.get(NUM, vocab_words[UNK]),
                        vocab_pos[pos]))
                else :
                    if lowercase:
                        t.append(vocab_tags[tag])
                        cwp.append(( word_to_char(word,vocab_chars), 
                            vocab_words.get(word.lower(), vocab_words[UNK]),vocab_pos[pos]))
                            #vocab_pos.get(pos, vocab_pos['X']) ))
            else :
                if lowercase:
                    t.append(vocab_tags[tag])
                    cwp.append((vocab_words.get(word.lower(), vocab_words[UNK]),vocab_pos[pos]))
                        #vocab_pos.get(pos, vocab_pos['X']) ))
        vec_words.append(cwp)
        vec_tags.append(t)
    return vec_words, vec_tags


    '''
    if config.shuffle_data:
        vec_words_shuffle = []
        vec_tags_shuffle = []
        idx = range(0, len(vec_words))
        shuffle(idx)
        for index in idx:
            vec_words_shuffle.append(vec_words[index])
            vec_tags_shuffle.append(vec_tags[index])
        vec_words = []
        vec_tags = []
        return vec_words_shuffle, vec_tags_shuffle
    '''
    
    

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags, config):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = config.padding_value
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def load_pkl_file(filename):
    with open(filename, 'rb') as fp:
        pkl_file = pickle.load(fp)
    return pkl_file

def save_pkl_file(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
    return
 