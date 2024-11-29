from scripts.main_scripts.run_cosine_test import full_cosine_test
from scripts.main_scripts.run_embeddings_extraction import full_embs_extraction
from scripts.main_scripts.calculate_edit_distance import edit_distance
from scripts.main_scripts.extract_ipas import extract_ipas
from phon_utility.full_pipeline_utils import (assign_correct_dataset_to_correct_embs, 
                                                clean_file, 
                                                articulatory_embs_format_correction,
                                                check_errors_in_args,
                                                contains_sentence)
import subprocess
from os.path import join
from os import listdir
from data.datasets.from_csv_to_txt_for_arpabet_transcript import from_csv_to_txt_for_arpabet_transcript
import warnings
import argparse

# We cannot include the arpabet transcriber atm. You should add here your ARPA transcriber
ARPABET_TRANSCRIBER_PATH = 'text_to_ARPABET/convert_arpabet.py'


def main():
    # Extract IPA transcriptions
    parser = argparse.ArgumentParser(description='Full test pipeline for phonetic embeddings.')
    parser.add_argument('--clean_dataset_paths', nargs='+', help='List of paths to CSV files', required=True)
    parser.add_argument('--IPA_path', type=str, required=False, help='FOLDER path to save the definitive IPA and ARPABET transcriptions')
    parser.add_argument('--columns_order', nargs='+', default=['a', 'b', 'c'], help='List of columns in the order they appear in the CSV files')
    parser.add_argument('--ARPA_path', type=str, required=False, help='FOLDER path to save the definitive ARPABET transcriptions')
    parser.add_argument('--embeddings_path', type=str, required=True, help='Path to output directory.')
    parser.add_argument('--p2v_model', type=str, required=False, help='Path to the Phoneme2vec model.')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save the results.')
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size for the models.')
    parser.add_argument('--selected_models', type=str, nargs='+', required=True, help='Selected models to run.')
    parser.add_argument('--load_ipa_paths', type=str, nargs='+', required=False, help='Paths to IPA already extracted for the selected "grapheme" datasets.')
    parser.add_argument('--load_arpa_paths', type=str, nargs='+', required=False, help='Paths to ARPA already extracted for the selected "grapheme" datasets.')
    parser.add_argument('--skip_edit_distance_test', action='store_true', help='If enabled, this will skip the distance test.')
    parser.add_argument('--do_not_extract_embeddings', action='store_true', help='If enabled, this will skip the embeddings extraction phase and will go straight to the cosine test.')
    parser.add_argument('--articulatory_embs_do_not_need_format_correction', action='store_true', help='If enabled, this will skip the format correction for the articulatory embeddings.')
    args = parser.parse_args()

    check_errors_in_args(args)

    if args.load_ipa_paths:
        print('Loading IPA transcriptions...')
        ipa_paths = args.load_ipa_paths
    else:
        print('Extracting IPA transcriptions...')
        ipa_paths = extract_ipas(args.clean_dataset_paths, args.IPA_path, args.columns_order)

    if args.load_arpa_paths:
        print('Loading ARPA transcriptions...')
        arpa_paths = args.load_arpa_paths
    else:
        print('Extracting ARPABET transcriptions...')
        arpa_paths = []
        for clean_dataset in args.clean_dataset_paths:
            txt_file_path = join(args.ARPA_path, clean_dataset.split('/')[-1].split('.csv')[0] + '.txt')
            from_csv_to_txt_for_arpabet_transcript(clean_dataset, txt_file_path)
            subprocess.run(['python3', ARPABET_TRANSCRIBER_PATH, 
                            '--file', txt_file_path])
            arpa_paths.append(txt_file_path.replace('.txt', 'ARPA.txt'))
        
        for path in arpa_paths:
            clean_file(path)
        
    if args.skip_edit_distance_test:
        pass
    else:
        print('Calculating edit distance...')
        edit_distance(ipa_paths)

    # In the case you are not extracting the embeddings again (because you already did it and you just want to test); args.embeddings_path
    # Should be the path to a folder of the embeddings you already extracted and that you want to test. 
    if args.do_not_extract_embeddings:
        pass
    else:
        print('Running full embeddings extraction...')
        for clean_dataset, ipa_dataset, arpa_dataset in zip(args.clean_dataset_paths, ipa_paths, arpa_paths):
            args.ARPABET_words_path = arpa_dataset
            args.IPA_words_path = ipa_dataset
            args.NORMAL_words_path = clean_dataset
            full_embs_extraction(args)
    
    print('Running cosine similarity test...')
    full_embeddings = listdir(args.embeddings_path)

    articulatory_embs_paths = [contains_sentence('Articulatory', full_embedding) for full_embedding in full_embeddings]
    articulatory_embs_paths = [join(args.embeddings_path, full_embedding) for art, full_embedding in zip(articulatory_embs_paths, full_embeddings) if art is True]
    
    # Every time you train from scratch articulatory embs with the full pipeline or the extraction_phoentic_embs, they will need a very smooth
    # but necessary format adapation. In case you already trained the embs and already had the format adapation done, you must skip this phase,
    # simply putting the flag --articulatory_embs_do_not_need_format_correction.
    if args.articulatory_embs_do_not_need_format_correction or "ArticulatoryPhonemes" not in args.selected_models:
        pass
    else:
        if len(articulatory_embs_paths) == 0:
            warnings.warn("No articulatory embeddings found. Skipping articulatory embeddings format correction. Did you use the correct name? Are you excluding them voluntary?")
        else:
            for art_embs_path, ipa_dataset_path, clean_dataset_path in zip(articulatory_embs_paths, ipa_paths, args.clean_dataset_paths):
                articulatory_embs_format_correction(embs_dict_path=art_embs_path, 
                                                ipa_dataset=ipa_dataset_path, 
                                                grapheme_dataset=clean_dataset_path, 
                                                output_path=art_embs_path)

    clean_dataset_paths, full_embeddings = assign_correct_dataset_to_correct_embs(args.clean_dataset_paths, full_embeddings)
    full_cosine_test(clean_dataset_paths,
                    [join(args.embeddings_path, full_embedding) for full_embedding in full_embeddings],  
                    [join(args.results_path, full_embedding.split('.')[0] + '_cosine.csv') for full_embedding in full_embeddings])

if __name__ == "__main__":
    main()