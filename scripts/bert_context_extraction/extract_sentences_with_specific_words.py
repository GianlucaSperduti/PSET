from datasets import load_dataset
from phon_utility.save_and_load import PickleLoader, PickleSaver
import re
import os
from tqdm import tqdm
import argparse


def select_substrings(text_list, word, n = 5):
        word_regex = r'\b' + re.escape(word.lower()) + r'\b'
        matches = []
        for text in text_list:
            if re.search(word_regex, text):
                matches.append(text)
                if len(matches) == n:
                    break
        if len(matches) < n:
            print(f'Word {word} not found {n} times in the text')
        return matches

def extract_sentences_with_word(word_list_path, sentences_list, n = 5):
    training_dict = {}
    with open(word_list_path, 'r') as f:
        word_list = f.readlines()
        word_list = [word.strip() for word in word_list]
    for word in tqdm(word_list):
        sentence_selection = select_substrings(sentences_list, word, n = n)
        training_dict[word] = sentence_selection
    return training_dict

# This code is needed if we want to extract embeddings for contextual models for words that are not in the corpus (if we don't want to search in other corpuses)
def fill_empty_items_with_sentence(dictionary, sentence):
    for _, item in dictionary.items():
        if not item:
            sentence = sentence.replace('()', _)
            dictionary[_] = [sentence]
    return dictionary

def check_len_items(w_to_s: dict):
    items_to_solve = []
    for key, item in w_to_s.items():
        if len(item) < 10:
            items_to_solve += (key, len(item))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract sentences with specific words')
    parser.add_argument('--word_list_path', type=str, help='Path to the word list file')
    # Id useful to recognize the extraction process. will be used to tie the primary 
    # extraction file with the integrations (in case sentences are missing and need to be entered manually)
    parser.add_argument('--id', default= 0, type=str, help='Identifier for tracking the extraction process.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to load')
    parser.add_argument('--split', type=str, default='train', help='Split of the dataset to use')
    parser.add_argument('--n', type=int, default=10, help='Number of sentences to extract per word')
    parser.add_argument('--log_path', type=str, help='Path to the log file')
    parser.add_argument('--save_path', type=str, help='Path to save the extracted sentences')

    args = parser.parse_args()

    word_list_path = args.word_list_path
    dataset = args.dataset
    split = args.split
    n = args.n
    log_path = args.log_path.split('.')[0] + f'_{args.id}.' + args.log_path.split('.')[1]
    save_path = args.save_path.split('.')[0] + f'_{args.id}.' + args.save_path.split('.')[1]

    train_dataset = load_dataset(dataset, split)

    sentences = extract_sentences_with_word(word_list_path, train_dataset['train']['text'], n=n)
    
    # A sort of warnings here.
    # Decide if you want to proceed for a fast test with a few sentences or if you want to extract more sentences
    
    check_len_items(sentences)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        for key, item in sentences.items():
            f.write(f'{key},{len(item)}\n')

    # Very probable: here it will be better a sort of txt saver, not a PickleSaver...
    PickleSaver.save(sentences, save_path)
