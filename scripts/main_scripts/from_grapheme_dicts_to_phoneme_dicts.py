import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Takes in input a grapheme based words to sentences dict, transforms it into a phonemes to phonetic_sentences dict.')
    parser.add_argument('--dictionary_path', type=str, help='Path to the dictionary file.')
    parser.add_argument('--plain_txt_dict_path', type=str, help='Path to the final file.')
    parser.add_argument('--transcriptions_path', type=str, help='Path to the transcriptions file.')
    parser.add_argument('--path_final_transcribed_dictionary', type=str, help='Path to the final file.')
    args = parser.parse_args()

    processes = []
    from_dict_to_sentences = subprocess.Popen(['python', 'scripts/from_dict_to_sentences.py',
                                '--dictionary_path', args.dictionary_path,
                                '--plain_txt_dict_path', args.plain_txt_dict_path],
                                stdout=subprocess.PIPE)
    processes.append(from_dict_to_sentences)
                                
    from_sentences_to_ipa = subprocess.Popen(['python', 'scripts/extract_ipa.py',
                                               "--txt_clean_path", args.plain_txt_dict_path,
                                               "--transcription_path", args.transcriptions_path],)
    processes.append(from_sentences_to_ipa)

    from_ipa_sentences_to_dict = subprocess.Popen(['python', 'scripts/from_sentences_to_dict.py',
                                         '--dictionary_path', args.dictionary_path,
                                         '--transcriptions_path', args.transcriptions_path,
                                         '--path_final_transcribed_dictionary', args.path_final_transcribed_dictionary])
    processes.append(from_ipa_sentences_to_dict)    

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == '__main__':
    main()