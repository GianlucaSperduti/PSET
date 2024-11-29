from typing import final
import pandas as pd
import subprocess
import os
import glob
import argparse

def remove_tmp(if_files):
    for file in if_files:
        if '_final_IPA' not in file:
            os.remove(file)

def extract_ipas(paths, definitive_output_path, columns_order):
    lists_of_IPA_files = []
    for p in paths: 
        print(p)
        lang = 'en-us'
        if 'chinese' in p:
            lang = 'zho-t'
        tmp_df = pd.read_csv(p)
        os.makedirs('tmp', exist_ok=True)
        if_files = glob.glob('tmp/*')
        remove_tmp(if_files)
        for column in tmp_df:
            tmp_path = f'tmp/tmp_{str(column)}'
            with open(f'{tmp_path}.csv', 'w') as f:
                f.write("\n".join(tmp_df[column].astype(str))) 
            process = subprocess.run(['python', 'phonetic_embeddings/scripts/extract_ipa.py', 
                                        '--txt_clean_path', f'{tmp_path}.csv', 
                                        '--transcription_path', f'{tmp_path}_IPA.txt',
                                        '--lang', lang],
                                         stdout=subprocess.PIPE)

        

        final_dataframe = pd.DataFrame()
        list_of_tmp_files = os.listdir('tmp')
        for column in list_of_tmp_files:
            if 'IPA' in column:
                df = pd.read_csv(os.path.join('tmp', column), header=None)
                for each_transcription in df.iterrows():
                    df.iloc[each_transcription[0]] = each_transcription[1][0].split('|')[1]
                final_dataframe[column.split('_')[1].split('_')[0]] = df[0]

        final_dataframe = final_dataframe.reindex(columns=columns_order)
        final_dataframe_path = f'{definitive_output_path}/_{p.split("/")[-1].split(".csv")[0]}_final_IPA.csv'               
        final_dataframe.to_csv(final_dataframe_path, index=False)
        lists_of_IPA_files.append(final_dataframe_path)

    remove_tmp(if_files)
    return lists_of_IPA_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process phonetic transcriptions.')
    parser.add_argument('--clean_dataset_paths', nargs='+', help='List of paths to CSV files', required=True)
    parser.add_argument('--output_path', type=str, required=True, help='Path to the FOLDER to save the definitive IPA transcriptions')
    parser.add_argument('--columns_order', nargs='+', default=['a', 'b', 'c'], help='List of columns in the order they appear in the CSV files')
    args = parser.parse_args()
    
    extract_ipas(args.clean_dataset_paths, args.output_path, args.columns_order)
