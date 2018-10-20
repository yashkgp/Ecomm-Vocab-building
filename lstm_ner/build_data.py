import numpy as np
from config import Config
# Special Tokens
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "$NONE$"


def load_dataset(filename, lowercase= False):
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
    with open(filename) as fp:
        sent_words, sent_tags, sent_pos = set(), set() ,set()
        for line in fp:
            line = line.strip()
            if (len(line) == 0):
                # Empty line and also end of a sentence
                if len(sent_words) != 0 or line.startswith("-DOCSTART"):
                    pass
            else :
                if len(line) < 2:
                    continue
                line_content = line.split("\t")
                word = line_content[0] 
                if lowercase :
                    word =word.lower()
                tag =line_content[1]
                pos = line_content[2]
                sent_words.add(word)
                sent_tags.add(tag)
                sent_pos.add(pos)
    print "Done Reading dataset from file : " + filename    
    return sent_words, sent_tags, sent_pos


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """Writes each word from vocabs to a file
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))

def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in numpy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words in dataset:
        for word in words:
            vocab_char.update(word)
    return vocab_char


def main():
    config = Config(phase="train")

    train_words, train_tags, train_pos = load_dataset(config.filename_train, lowercase= True)
    dev_words, dev_tags, dev_pos   = load_dataset(config.filename_dev, lowercase= True)
    test_words, test_tags, test_pos  = load_dataset(config.filename_test, lowercase= True)

    word_set = train_words.union(dev_words).union(test_words)
    tag_set = train_tags.union(dev_tags).union(test_tags)
    pos_set = train_pos.union(dev_pos).union(test_pos)

    vocab_glove = get_glove_vocab(config.filename_glove)
    vocab = word_set & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    write_vocab(vocab, config.filename_words)
    write_vocab(tag_set, config.filename_tags)
    write_vocab(pos_set, config.filename_pos )

    # Trim embedding Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    vocab_chars = get_char_vocab(train_words)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
