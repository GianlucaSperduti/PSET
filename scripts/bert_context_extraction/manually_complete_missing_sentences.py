import argparse
from phon_utility.save_and_load import PickleLoader, PickleSaver
from os.path import join
import os

def read_log(log_path):
    with open(log_path, 'r') as f:
        log_lines = f.readlines()
        log_lines = [line.split(',') for line in log_lines]
        log_lines = [(line[0], int(line[1].strip())) for line in log_lines]
        return log_lines

def find_words_to_adjust(logs, log_to_check):
    words = []
    for log in log_to_check:
            log_lines = read_log(join(logs, log))
            for line in log_lines:
                if line[1] < 10:
                    words.append((line[0], line[1]))
    return words


def parse_file_to_dict(file_path, keys_list):
    data_dict = {key: [] for key in keys_list}
    current_key = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Remove trailing comma if present
            if line.endswith(','):
                line = line[:-1].strip()

            if line in data_dict:
                # Found a key, initialize it
                current_key = line
            elif current_key:
                # Add sentences to the current key
                data_dict[current_key].append(line)
                
    return data_dict

def main():
    parser = argparse.ArgumentParser(description='Manually complete missing sentences')
    parser.add_argument('--contextual_emb_folder', type=str, help='Path to the word list file')
    parser.add_argument('--id', default= 0, type=str, help='Identifier for tracking the extraction process.')
    parser.add_argument('--prepare', action='store_true', help='it prepares the file in which the sentences will be added.')
    parser.add_argument('--load_and_export', action='store_true', help='it loads the file with the new sentences and exports the new embeddings.')
    args = parser.parse_args()

    if args.prepare and args.load_and_export:
        raise ValueError('You can only choose one between prepare and load_and_export.')

    # List all the files in the two main directories
    logs_path = join(args.contextual_emb_folder, 'logs')
    files_in_logs = os.listdir(logs_path)
    log_to_check = [file for file in files_in_logs if str(args.id) in file]
    words_to_adjust = find_words_to_adjust(logs_path, log_to_check)
    sentences_path = join(args.contextual_emb_folder, 'w_to_s')
    adding_sentences = join(args.contextual_emb_folder, 'adding_sentences')
    os.makedirs(adding_sentences, exist_ok=True)
    final_file_path = join(adding_sentences, str(args.id) + '.txt')

    if args.prepare:    
        if os.path.exists(final_file_path):
            raise ValueError(f'The file {str(args.id)} already exists. Please check the file before running the script.')

        
        log_to_check = [file for file in files_in_logs if str(args.id) in file]
        if len(log_to_check) > 1:
            raise ValueError('There are more than one log file with the selected id. Please check the logs before running the script.')

        if len(log_to_check) == 0:
            raise ValueError('There are no logs with the selected id. Please check the logs before running the script.')
        
        with open(final_file_path, 'w') as f:
            for word in words_to_adjust:
                f.write(f'{word[0], word[1]}\n')
    
    if args.load_and_export:
        if not os.path.exists(final_file_path):
            raise ValueError(f'The file {str(args.id)} does not exist. Please check the file before running the script.')
        
        files_in_sentences = os.listdir(sentences_path)
        sentence_to_check = [file for file in files_in_sentences if str(args.id) in file]
        if len(sentence_to_check) > 1:
            raise ValueError('There are more than one sentences file with the selected id. Please check the sentences before running the script.')

        dict_sentences = PickleLoader.load(join(sentences_path, sentence_to_check[0]))

        words_to_adjust_exporting_phase = [word[0] for word in words_to_adjust]
        dictionary_to_append = parse_file_to_dict(join(adding_sentences, str(args.id) + '.txt'), words_to_adjust_exporting_phase)

        for key, item in dict_sentences.items():
            if key in words_to_adjust_exporting_phase:
                dict_sentences[key] += dictionary_to_append[key]

        PickleSaver.save(dict_sentences, join(adding_sentences, str(str(args.id) + 'COMPLETE.pkl')))


if __name__ == '__main__':
    main()

        
