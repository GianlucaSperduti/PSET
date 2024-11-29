import argparse
from scripts import cosine_sim_test

def full_cosine_test(dataset_paths, embeddings_paths, output_paths):
    for dataset_path, embeddings_path, output_path in zip(dataset_paths, embeddings_paths, output_paths):
        cosine_sim_test.main(dataset_path, embeddings_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate cosine similarities for multiple datasets.')
    parser.add_argument('--dataset_paths', type=str, nargs='+', help='Paths to the CSV files containing the datasets.')
    parser.add_argument('--embeddings_paths', type=str, nargs='+', help='Paths to the files containing the extracted embeddings.')
    parser.add_argument('--output_paths', type=str, nargs='+', help='Paths to save the output CSV files with cosine similarities.')

    args = parser.parse_args()
    
    if len(args.dataset_paths) != len(args.embeddings_paths) or len(args.dataset_paths) != len(args.output_paths):
        parser.error("The number of dataset paths, embeddings paths, and output paths must be the same.")

    full_cosine_test(args.dataset_paths, args.embeddings_paths, args.output_paths)
