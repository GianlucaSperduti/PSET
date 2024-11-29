import argparse
from phon_utility.phon_utility import PathChecker
from phon_utility.save_and_load import PickleLoader, PickleSaver
from data.data_source import CMUdictionary2Vec, PWEsuite4p2v
from embeddings.embeddings_trainer import Phoneme2VecTrainer


def load_or_save(loader, saver, path, default_object):
    return [loader.load(path) if PathChecker().check(path) else saver.save(default_object, path)]

def main():
    parser = argparse.ArgumentParser(description='Phoneme2Vec Trainer and Loader')
    parser.add_argument('--dict_path', type=str, default='phonetic_embeddings/data/datasets/pwedictionary2vec.pkl', help='Path to CMU dictionary file')
    parser.add_argument('--model_path', type=str, default='trained_models/word2vec/PWE2Vec_trained.pkl', help='Path to model file')
    parser.add_argument('--dataset', type=str, default='pwe', help='Training dataset: for now, only cmu and pwe available.')
    args = parser.parse_args()

    loader = PickleLoader()
    saver = PickleSaver()

    dataset_list = {'cmu':CMUdictionary2Vec(),
               'pwe':PWEsuite4p2v()} 
    
    dataset = dataset_list[args.dataset]
    
    # Load or save dataset dictionary
    dataset_dictionary = load_or_save(loader, saver, args.dict_path, dataset)

    # Load or train Phoneme2Vec dataset model
    Phoneme2Vec_dataset = load_or_save(loader, saver, args.model_path,
                                   Phoneme2VecTrainer(dataset_dictionary[0].dataset).train(return_model=True))

    print('test')


if __name__ == '__main__':
    main()
