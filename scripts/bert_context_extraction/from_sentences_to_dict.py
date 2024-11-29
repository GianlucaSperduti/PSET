from phon_utility.save_and_load import PickleLoader, PickleSaver
import argparse

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

def main():
    parser = argparse.ArgumentParser(description='From dictionary to sentences')
    parser.add_argument('--dictionary_path', type=str, help='Path to the dictionary file.')
    parser.add_argument('--transcriptions_path', type=str, help='Path to the transcriptions file.')
    parser.add_argument('--path_final_transcribed_dictionary', type=str, help='Path to the final file.')
    args = parser.parse_args()

    l_to_tr = parse_file(args.transcriptions_path)
    dictionary = PickleLoader.load(args.dictionary_path)
    result = phon_dict_creator(dictionary, l_to_tr)
    PickleSaver.save(result, args.path_final_transcribed_dictionary)

if __name__=='__main__':
    main()
    
    
    