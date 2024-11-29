from phon_utility.save_and_load import PickleSaver
import pandas as pd

# This is a file that must take as input the transcriptions made by the infer dataset function, take
# the homophones, anchors, synonyms .pkl file that contains the dictionary with the contextual sentences, and create a dictionary
# with the corresponding sentences for each word, but phonetically transcribed. This dictionary will then be saved in a .pkl file

def parse_file(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if '|' in line:
                key, value = line.split('|')
                value = value.replace('{', '').replace('}', '')
                result_dict[key.strip()] = value.strip()
    return result_dict

def phon_dict_creator(h_a_s: dict, latin_to_trans:str):
    for word, context_list in h_a_s.items():
        cont_list_trans = []
        for context in context_list:   
            for sentence, transcription in latin_to_trans.items():
                if context.strip() == sentence.strip():
                    cont_list_trans.append(transcription)
            h_a_s[word] = cont_list_trans
            h_a_s_new = h_a_s
    return h_a_s_new

def arpabet_dict_creator(h_a_s: dict, original_dict:dict):
    filtered_dict = {}
    for word, arpa_transcription in h_a_s.items():
        for exact_word in list(original_dict):
            if word == exact_word:
                filtered_dict[word] = arpa_transcription
    
    # Print the difference in keys between filtered_dict and h_a_s
    print('Length of filtered_dict:', len(filtered_dict))
    print('Length of original_dict:', len(set(list(original_dict))))
    for element in set(list(original_dict)):
        if element not in filtered_dict.keys():
            print(element)

    return filtered_dict

if __name__=='__main__':
    a_h_s = 'homophones'
    ARPABET_path = f'../phonetic_embeddings/text_to_ARPABET/transcriptions/homophone_testARPA.txt'
    csv_path = f'../phonetic_embeddings/data/datasets/SemPhonDiscr/Homophone_GOLD_STANDARD.csv'
    original_test_dictionary = pd.read_csv(csv_path)
    arpabet_parsed = parse_file(ARPABET_path)
    result = arpabet_dict_creator(arpabet_parsed, original_test_dictionary[a_h_s])
    PickleSaver.save(result, f'../phonetic_embeddings/data/datasets/SemPhonDiscr/single_words/phonetic_words/{a_h_s}_ARPABET.pkl')