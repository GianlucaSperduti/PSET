from os import name
from phon_utility.save_and_load import PickleLoader, PickleSaver
import argparse
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", help="Path to the 3 trained embeddings.pkl")
    parser.add_argument("--save_path", help="Path to save the aggregated embeddings")
    parser.add_argument("--file_name", default='aggregated', help="Name of the file to save the aggregated embeddings")
    args = parser.parse_args()

    anchor = PickleLoader.load(join(args.embeddings_path,'anchors.pkl'))
    homophones = PickleLoader.load(join(args.embeddings_path,'homophones.pkl'))
    synonyms = PickleLoader.load(join(args.embeddings_path,'synonyms.pkl'))

    # Aggregate the embeddings into a single flat dictionary
    aggregated_embeddings = {}
    aggregated_embeddings.update(anchor)
    aggregated_embeddings.update(homophones)
    aggregated_embeddings.update(synonyms)

    PickleSaver.save(aggregated_embeddings, join(args.save_path, args.file_name + '.pkl'))

if __name__ == '__main__':
    main()