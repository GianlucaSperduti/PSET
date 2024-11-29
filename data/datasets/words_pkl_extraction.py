from phon_utility.save_and_load import PickleSaver
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="CSV file with homophones/spl_vars, synonyms and anchors")
    parser.add_argument("--save_path", help="Path to save the pickled file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    df = df.fillna('')
    homophones = df['homophones'].to_list()
    synonyms = df['synonyms'].to_list()
    anchors = df['anchors'].to_list()

    PickleSaver.save(homophones, os.path.join(args.save_path, 'homophones.pkl'))
    PickleSaver.save(synonyms, os.path.join(args.save_path, 'synonyms.pkl'))
    PickleSaver.save(anchors, os.path.join(args.save_path, 'anchors.pkl'))

    print('Operation completed successfully!')

if __name__ == '__main__':
    main()


