import argparse
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run phonetic and semantic models for extracting embeddings.')

    # Paths for data files
    parser.add_argument('--ARPABET_words_path', type=str, required=True, help='Path to ARPABET words data file.')
    parser.add_argument('--IPA_words_path', type=str, required=True, help='Path to IPA words data file.')
    parser.add_argument('--NORMAL_words_path', type=str, required=True, help='Path to normal words data file.')
    parser.add_argument('--w_t_sentences_path', type=str, default='', help='Path to word-token sentences data file.')
    parser.add_argument('--w_t_IPA_sentences_path', type=str, default='', help='Path to word-token IPA sentences data file.')
    parser.add_argument('--embeddings_path', type=str, required=True, help='Path to output directory.')
    parser.add_argument('--p2v_model', type=str, required=True, help='Path to the Phoneme2vec model.')
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size for the models.')
    parser.add_argument('--selected_models', type=str, nargs='+', required=True, help='Selected models to run.')

    return parser.parse_args()

def full_embs_extraction(args):

    selected_models = args.selected_models

    # Experiment dictionary
    semantic_models = {
        'Word2Vec': {
            'file_path': args.NORMAL_words_path,
            'secondary_phonetic_path': args.IPA_words_path,
            'embeddings_path': args.embeddings_path,
            'batch_size': args.batch_size
        },
        'ClassicBert': {
            'file_path': args.NORMAL_words_path,
            'secondary_phonetic_path': args.IPA_words_path,
            'embeddings_path': args.embeddings_path,
            'batch_size': args.batch_size
        }
    }

    phonetic_models = {
        'Phoneme2Vec': {
            'secondary_phonetic_path': '',
            'file_path': args.ARPABET_words_path,
            'embeddings_path': args.embeddings_path,
            'p2v_model': args.p2v_model,
            'batch_size': args.batch_size
        },
        'XPhoneBERT': {
            'file_path': args.IPA_words_path,
            'secondary_phonetic_path': args.IPA_words_path,
            'embeddings_path': args.embeddings_path,
            'p2v_model': '',
            'batch_size': args.batch_size
        },
        'ArticulatoryPhonemes': {
            'file_path': args.IPA_words_path,
            'secondary_phonetic_path': args.IPA_words_path,
            'embeddings_path': args.embeddings_path,
            'p2v_model': '',
            'batch_size': args.batch_size
        }
    }


    processes = []

    # Running the experiments for semantic models
    for model_name, model_args in semantic_models.items():
        if model_name in selected_models:
            process = subprocess.Popen([
                'python', 'phonetic_embeddings/scripts/embedding_extraction/extract_embs.py',
                '--dataset', 'SemanticDataset',
                '--file_path', model_args['file_path'],
                '--secondary_phonetic_path', model_args['secondary_phonetic_path'],
                '--embeddings_path', model_args['embeddings_path'],
                '--model', model_name,
                '--load_last_batch', 'false',
                '--batch_size', model_args['batch_size']
            ])
            processes.append(process)

    # Running the experiments for phonetic models
    for model_name, model_args in phonetic_models.items():
        if model_name in selected_models:
            process = subprocess.Popen([
                'python', 'phonetic_embeddings/scripts/embedding_extraction/extract_phonetic_embs.py',
                '--dataset', 'PhoneticDataset',
                '--file_path', model_args['file_path'],
                '--secondary_phonetic_path', model_args['secondary_phonetic_path'],
                '--embeddings_path', model_args['embeddings_path'],
                '--phonetic_model', model_name,
                '--p2v_model', model_args['p2v_model'],
                '--load_last_batch', 'false',
                '--batch_size', model_args['batch_size']
            ])
            processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()

if __name__ == "__main__":
    args = parse_arguments()
    full_embs_extraction(args)
