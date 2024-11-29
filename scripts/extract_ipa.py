from ipa_extraction.IpaExtractor import IpaTranscriptionSentence, IpaTranscription
from phon_utility.save_and_load import PickleLoader, PickleLoader, PickleSaver
import os
import glob
import argparse

def get_phonetic_transcriptions_from_sentence_dict(sentence_dict, transcription_path:str, batch_size:int=4):
    batch_count = 0
    transcription_dict = {}
    for key, item in sentence_dict.items():
        batch_count += 1
        sentence_list = IpaTranscriptionSentence.generate_phonetic_transcriptions(item, language='en-us') 
        transcription_dict[key] = sentence_list
        if batch_count % batch_size == 0:
            path = os.path.join(transcription_path, str(batch_count) + '.pkl')
            PickleSaver.save(transcription_dict, path)
            transcription_dict = {}

def get_phonetic_transcriptions_from_txt(original_path, transcription_path:str, lang:str, batch_size:int=64):
    sentence_list = IpaTranscription.generate_phonetic_transcriptions(original_path, transcription_path, batch_size, language=lang) 

def save_final_transcription(transcription_path:str, file_name:str='final'):
    # It should load all the pkl in the folder and merge them in a single file
    embeddings = {}
    for file in glob.glob(os.path.join(transcription_path, '*.pkl')):
        embs_dict = PickleLoader.load(file)
        embeddings.update(embs_dict)
    
    merged_path = os.path.join(transcription_path, f'{file_name}.pkl')
    PickleSaver.save(embeddings, merged_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors_path", default="none", help="Path to the anchors file")
    parser.add_argument("--homophones_path", default="none", help="Path to the homophones file")
    parser.add_argument("--synonyms_path", default="none", help="Path to the synonyms file")
    parser.add_argument("--txt_clean_path", default="none", help="Path to the txt_clean file")
    parser.add_argument("--transcription_path", help="Path to save the final path for the transcriptions")
    parser.add_argument("--lang", default="en-us", help="Language for the phonetic transcriptions")
    args = parser.parse_args()

    if args.anchors_path != 'none' and args.homophones_path != 'none' and args.synonyms_path != 'none':
        anchors = PickleLoader.load(args.anchors_path)
        homophones = PickleLoader.load(args.homophones_path)
        synonyms = PickleLoader.load(args.synonyms_path)
        anchors = get_phonetic_transcriptions_from_sentence_dict(anchors, os.path.join(args.transcription_path, 'anchors'))
        homophones = get_phonetic_transcriptions_from_sentence_dict(homophones, os.path.join(args.transcription_path, 'homophones'))
        synonyms = get_phonetic_transcriptions_from_sentence_dict(synonyms, os.path.join(args.transcription_path, 'synonyms'))
        save_final_transcription(os.path.join(args.transcription_path, 'anchors'), file_name='anchors')
        save_final_transcription(os.path.join(args.transcription_path, 'homophones'), file_name='homophones')
        save_final_transcription(os.path.join(args.transcription_path, 'synonyms'), file_name='synonyms')
    else:
        if args.txt_clean_path == 'none':
            raise ValueError('You must provide a path for either the txt_clean file, or the anchors, homophones and synonyms files.')
        get_phonetic_transcriptions_from_txt(args.txt_clean_path, args.transcription_path, lang=args.lang)

if __name__ == '__main__':
    main()