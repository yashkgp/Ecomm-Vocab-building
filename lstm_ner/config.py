import os
import datetime
import numpy as np
from log_utils import get_logger


def load_embeddings(filename):
    """
    Args: filename: path to the npz file
    Returns: matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        print "ERROR: Unable to locate file {}".format(filename)


def load_vocab(filename):
    """Loads vocab from a file
    Args: filename: (string) the format of the file must be one word per line.
    Returns:  d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
    except IOError:
        raise "ERROR: Unable to locate file {}".format(filename)
    return d

class Config():
    def __init__(self, phase="none"):

        if phase == "none":
            
            # Only using Configuration variables and vocabs

            self.vocab_words = load_vocab(self.filename_words)
            self.vocab_tags  = load_vocab(self.filename_tags)
            self.vocab_chars = load_vocab(self.filename_chars)
            self.vocab_pos = load_vocab(self.filename_pos)

        if phase == "train":
            
            self.dir_model = self.dir_model+"/"
            #self.dir_model = self.dir_model + str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")) + "/"
            
            # For trainng phase
            # directory for training outputs
            if not os.path.exists(self.dir_model):
                os.makedirs(self.dir_model)
    
            # Directory inside dir_model to save tf model
            self.model_dir = self.dir_model + "model"
    
    
            # create instance of logger
            self.logger = get_logger(self.dir_model + self.log_file)
            self.load_self()

        if phase == "restore":
            self.logger = None
            self.model_dir = self.dir_model + "model"
            self.load_self()


    def load_self(self):
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        self.vocab_pos = load_vocab(self.filename_pos)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)
        self.npos      = len(self.vocab_pos)

         # 2. get pre-trained embeddings
        self.embeddings = (load_embeddings(self.filename_trimmed))
            #if self.use_pretrained else None)        

    UNK = "$UNK$"
    NUM = "$NUM$"
    NONE = "$NONE$"

    special_tokens = [UNK, NUM, NONE]

    dir_output = "results/"
    dir_model  = dir_output + "model_"
    log_file   = "log.txt"

    # vocab (dictionaries saved as pkl files)
    filename_words = "data/vocab.txt"
    filename_chars = "data/chars.txt"
    filename_tags = "data/tags.txt"
    filename_pos = "data/pos.txt"
    #word embedding dimension is (nwords, dim_word)
    dim_word = 50
    # Word Embeddings pre-trained  
    # glove files
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True


    # dataset
    filename_dev = "data/valSmall"
    filename_test = "data/testSmall"
    filename_train = "data/trainSmall"

    padding_value = 0

    # Training Config variables
    retrain_embeddings = False
    use_word_embeddings = True
    nepochs          = 5
    batch_size       = 32 
    nepoch_no_imprv  = 5

    #Learning method
    learning_method  = "adam"
    default_lr       = 0.001
    learning_rates_list = [0.001]
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    
    # Dropout
    use_dropout = True
    dropout_rate = 0.5


    # Model type
    #biLSTM variable
    use_bilstm = True
    use_pos = False
    hidden_size_lstm = 30

    # Char Embedding variables
    dim_char = 30
    hidden_size_char = 50

    use_crf = False 
    use_chars = False 

    print "PARAMETER values \n"

    print "use_bilstm\t{}".format(use_bilstm) 
    print "use_crf\t{}".format(use_crf) 
    print "use_chars\t{}".format(use_chars) 

    print "nepochs\t{}".format(nepochs) 
    print "batch_size\t{}".format(batch_size) 
    print "nepoch_no_imprv\t{}".format(nepoch_no_imprv) 

    print "learning_method\t{}".format(learning_method) 
    print "default_lr\t{}".format(default_lr) 
    print "lr_decay\t{}".format(lr_decay) 
    print "clip\t{}".format(clip) 

    print "use_dropout\t{}".format(use_dropout) 
    print "dropout_rate\t{}".format(dropout_rate) 

    print "dim_char\t{}".format(dim_char) 
    print "hidden_size_char\t{}".format(hidden_size_char) 
    print "hidden_size_lstm\t{}".format(hidden_size_lstm) 

    print "shuffle_data\t{}".format(shuffle_data) 

