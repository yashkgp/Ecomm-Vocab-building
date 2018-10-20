from utils import load_dataset, vectorize_data
from seq_tag_lstm import NerLstm
from config import Config

def main():
    # create instance of config
    config = Config(phase ="train")

    # build model
    model = NerLstm(config)
    #model.build_model()
    print config.dir_model
    model.restore_session(config.dir_model)#+'/model.ckpt')

    # create dataset
    test_data = load_dataset(config.filename_test)

    test_vectorize = vectorize_data(test_data, config.vocab_words, 
        config.vocab_tags, config.vocab_pos, config.vocab_chars, config, lowercase =True )

    metrics = model.evaluate(test_vectorize,is_test_set=1)
    print metrics


if __name__ == "__main__":
    main()
