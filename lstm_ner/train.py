from utils import load_dataset, vectorize_data
from seq_tag_lstm import NerLstm
from config import Config

def main():

    #print "Running the Code for run number : " + str(run_number)

    config = Config(phase="train")

    # Load data from txt files
    train_data = load_dataset(config.filename_train)
    dev_data = load_dataset(config.filename_dev)
    #test_data = load_dataset(config.filename_test)

    #train_vectorize = vectorize_text_data(train_data, config.vocab_words, 
    #    config.vocab_chars, config.vocab_tags, config)
    train_vectorize = vectorize_data(train_data, config.vocab_words, 
        config.vocab_tags, config.vocab_pos, config.vocab_chars, config, lowercase =True )

    dev_vectorize = vectorize_data(dev_data, config.vocab_words, 
        config.vocab_tags, config.vocab_pos, config.vocab_chars, config, lowercase =True )


    #test_vectorize = vectorize_text_data(test_data, config.vocab_words, 
    #    config.vocab_chars, config.vocab_tags, config)

    model = NerLstm(config)
    model.build_model()
    model.train(train_vectorize, dev_vectorize, test_set=None)
