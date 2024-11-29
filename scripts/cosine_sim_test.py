import pandas as pd
import argparse
from data.data_source import CosineDatasetUtility
from SemPhonTest.CosineSimCalculation import CosineSim, CosineAnchorTest
from SemPhonTest.ScoreComparator import (ScoreComparator, 
                                         ScoreDifferenceFinder,
                                         ScoreComparatorFour,
                                         ScoreDifferenceFinderFour)
from phon_utility.save_and_load import PickleLoader

def main(dataset_path, embeddings_path, output_path):

    print(f'Calculating cosine similarities... for embeddings:', embeddings_path)

    pl = PickleLoader()
    
    dataset = pd.read_csv(dataset_path)
    extracted_embs = pl.load(embeddings_path)

    cosine_data_transformer = CosineDatasetUtility()
    transformed_dataset = cosine_data_transformer.apply(dataset)

    cosine_sim_calculator = CosineSim()
    cos_similarities = CosineAnchorTest().calc(transformed_dataset, extracted_embs, cosine_sim_calculator)

    if 'd' not in dataset:
        df_cos_similarities = CosineAnchorTest()._to_pandas(cos_similarities, columns=['a', 'a_score', 'b', 'b_score',
                                                                                    'c', 'c_score'])
        df_cos_similarities.to_csv(output_path)
        sc = ScoreComparator(df_cos_similarities)
        sd = ScoreDifferenceFinder(df_cos_similarities)
        scores = sc.compare_scores()
        top_differences = sd.find_top_differences(5)
        bottom_differences = sd.find_bottom_differences(5)
        with open(output_path + '_prevalences.txt', 'w') as f:
            f.write(f'b prevalence: {scores[0]}\n')
            f.write(f'c prevalence: {scores[1]}\n')
            f.write(f'Top differences: {top_differences}\n')
            f.write(f'Bottom differences: {bottom_differences}\n')
    else:
        df_cos_similarities = CosineAnchorTest()._to_pandas(cos_similarities, columns=['a', 'a_score', 'b', 'b_score',
                                                                                    'c', 'c_score', 'd', 'd_score'])
        df_cos_similarities.to_csv(output_path)
        nan_rows = df_cos_similarities[df_cos_similarities.isna().any(axis=1)]
        absent_rows = df_cos_similarities[df_cos_similarities['d_score'] == "absent"]
        if not nan_rows.empty or not absent_rows.empty:
            print(f'Warning: NaN values found in the dataset. Dropping rows with NaN values.')
            nan_rows.to_csv(output_path + '_nan_rows.csv')
            absent_rows.to_csv(output_path + '_absent_rows.csv')
            df_cos_similarities = df_cos_similarities.dropna()
            df_cos_similarities = df_cos_similarities[df_cos_similarities['d_score'] != "absent"]
        sc = ScoreComparatorFour(df_cos_similarities)
        sd = ScoreDifferenceFinderFour(df_cos_similarities)
        scores = sc.compare_scores()
        top_differences_b_c, top_differences_b_d = sd.find_top_differences(5)
        bottom_differences_c, bottom_differences_d = sd.find_bottom_differences(5)
        with open(output_path + '_prevalences.txt', 'w') as f:
            f.write(f'b prevalence: {scores[0]}\n')
            f.write(f'c prevalence: {scores[1]}\n')
            f.write(f'd prevalence: {scores[2]}\n')
            f.write(f'Top differences b-c: {top_differences_b_c}\n')
            f.write(f'Top differences b-d: {top_differences_b_d}\n')
            f.write(f'Bottom differences c: {bottom_differences_c}\n')
            f.write(f'Bottom differences d: {bottom_differences_d}\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate cosine similarities for a dataset.')
    parser.add_argument('--dataset_path', type=str, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--embeddings_path', type=str, help='Path to the file containing the extracted embeddings.')
    parser.add_argument('--output_path', type=str, help='Path to save the output CSV file with cosine similarities.')

    args = parser.parse_args()
    main(args.dataset_path, args.embeddings_path, args.output_path)
