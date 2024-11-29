from abc import abstractmethod, ABC
from os import stat
from typing import Type
import warnings

class ScoreUtilityInterface(ABC):
    @abstractmethod
    def clean_data(data):
        pass

class ScoreUtility(ScoreUtilityInterface):
    @staticmethod
    def is_string(value):
        return isinstance(value, str)

    @staticmethod
    def initialize_rows_to_drop(data):
            
        # Apply the function to both columns 'a' and 'b' and get boolean DataFrames
        is_string_a = data['a_score'].apply(ScoreUtility.is_string)
        is_string_b = data['b_score'].apply(ScoreUtility.is_string)
        is_string_c = data['c_score'].apply(ScoreUtility.is_string)

        # Combine the boolean DataFrames to identify rows where either column 'a' or 'b' is a string
        rows_to_drop = is_string_a | is_string_b | is_string_c

        return rows_to_drop
    
    @staticmethod
    def clean_data(data):
        rows_to_drop = ScoreUtility.initialize_rows_to_drop(data)
        # Drop those rows from the DataFrame
        data = data[~rows_to_drop]
        return data

class ScoreComparator:
    def __init__(self, data, score_utility_obj: Type[ScoreUtilityInterface] = ScoreUtility()):
        self.data = data
        self.score_utility_obj = score_utility_obj
        self.score_utility_obj.clean_data(self.data)
        self.total_rows = len(self.data)
        self.exclude_anchor_score()

    def exclude_anchor_score(self):
        # Exclude "anchor_score" column
        self.data = self.data.drop(columns=['a'], errors='ignore')

    def compare_scores(self):
        # Compare spl_var_score and synonym_score
        b_c_score = (self.data['b_score'] > self.data['c_score']).sum()
        c_b_score = (self.data['c_score'] > self.data['b_score']).sum()
        b_c_prevalence = b_c_score / self.total_rows
        c_b_prevalence = c_b_score / self.total_rows

        return b_c_prevalence, c_b_prevalence

class ScoreComparatorFour(ScoreComparator):
    def __init__(self, data, score_utility_obj: Type[ScoreUtilityInterface] = ScoreUtility()):
        super().__init__(data, score_utility_obj)
    
    def check_for_non_matching_scores(self):
        b_score_rows = (self.data['b_score'] > self.data['d_score']) & (self.data['b_score'] > self.data['c_score'])
        c_score_rows = (self.data['c_score'] > self.data['b_score']) & (self.data['c_score'] > self.data['d_score'])
        d_score_rows = (self.data['d_score'] > self.data['b_score']) & (self.data['d_score'] > self.data['c_score'])
        none_condition_met = ~(b_score_rows | c_score_rows | d_score_rows)
        non_condition = None
        n_non_matching_conditions = 0
        if none_condition_met.any():
            n_non_matching_conditions = len(self.data[none_condition_met])
            non_condition = self.data[none_condition_met]
        return non_condition, n_non_matching_conditions


    def compare_scores(self):
        if 'd_score' in self.data.columns:
            b_score = ((self.data['b_score'] > self.data['d_score']) & (self.data['b_score'] > self.data['c_score'])).sum()
            c_score = ((self.data['c_score'] > self.data['b_score']) & (self.data['c_score'] > self.data['d_score'])).sum()
            d_score = ((self.data['d_score'] > self.data['b_score']) & (self.data['d_score'] > self.data['c_score'])).sum()
            non_matching_condition, n_non_matching_conditions = self.check_for_non_matching_scores()
            if n_non_matching_conditions > 0:
                print(f'Non-matching conditions found: {n_non_matching_conditions}, rows: {non_matching_condition}')
                warnings.warn('Non-matching conditions found')
            self.total_rows -= n_non_matching_conditions
            b_prevalence = b_score / self.total_rows
            c_prevalence = c_score / self.total_rows
            d_prevalence = d_score / self.total_rows
            return b_prevalence, c_prevalence, d_prevalence
        else:
            warnings.warn('Column d_score not found in the dataset')
            b_c_prevalence, c_b_prevalence = super().compare_scores()
            return b_c_prevalence, c_b_prevalence

    
class ScoreDifferenceFinder:
    def __init__(self, data, score_utility_obj: Type[ScoreUtilityInterface] = ScoreUtility()):
        self.score_utility_obj = score_utility_obj
        self.data = self.score_utility_obj.clean_data(data)

    @staticmethod
    def find_absoulte_difference(data, column_name, column_a, column_b):
        # Calculate the absolute differences between spl_var_score and synonym_score
        data[column_name] = abs(data[column_a] - data[column_b])
        data[column_name] = data[column_name].astype(float)
        return data

    def find_top_differences(self, top_n=5):
        if 'score_difference_b_c' not in self.data.columns:
            self.data = self.find_absoulte_difference(self.data, 'score_difference_b_c', 'b_score', 'c_score')
        # Sort by the score_difference in descending order and take the top N differences
        top_differences_b_c = self.data.nlargest(top_n, 'score_difference_b_c')
        # Extract relevant columns for the top differences
        top_differences_b_c = top_differences_b_c[['b', 'c', 'b_score', 'c_score']]
        return top_differences_b_c
    
    def find_bottom_differences(self, bottom_n=5):
        if 'score_difference_b_c' not in self.data.columns:
            self.data = self.find_absoulte_difference(self.data, 'score_difference_b_c', 'b_score', 'c_score')
        # Sort by the score_difference in ascending order and take the bottom N differences
        bottom_differences_c = self.data.nsmallest(bottom_n, 'score_difference_b_c')
        # Extract relevant columns for the bottom differences
        bottom_differences_c = bottom_differences_c[['b', 'c', 'b_score', 'c_score']]
        return bottom_differences_c


class ScoreDifferenceFinderFour(ScoreDifferenceFinder):
    def __init__(self, data, score_utility_obj: Type[ScoreUtilityInterface] = ScoreUtility()):
        super().__init__(data, score_utility_obj)

    def find_top_differences(self, top_n=5):
        top_differences_b_c = super().find_top_differences(top_n)
        if 'd_score' in self.data.columns:
            if 'score_difference_b_d' not in self.data.columns:
                self.data = self.find_absoulte_difference(self.data, 'score_difference_b_d', 'b_score', 'd_score')
            top_differences_b_d = self.data.nlargest(top_n, 'score_difference_b_d')
            top_differences_b_d = top_differences_b_d[['b', 'd', 'b_score', 'd_score']]
            return top_differences_b_c, top_differences_b_d
        else:
            warnings.warn('Column d_score not found in the dataset')
            return top_differences_b_c
    
    def find_bottom_differences(self, bottom_n=5):
        if 'score_difference_b_c' not in self.data.columns:
            self.data = self.find_absoulte_difference(self.data, 'score_difference_b_c', 'b_score', 'c_score')
        if 'score_difference_b_d' not in self.data.columns:
            self.data = self.find_absoulte_difference(self.data, 'score_difference_b_d', 'b_score', 'd_score')

        if 'd_score' in self.data.columns:
            bottom_differences_c = super().find_bottom_differences(bottom_n)
            bottom_differences_d = self.data.nsmallest(bottom_n, 'score_difference_b_d')
            # Extract relevant columns for the bottom differences
            bottom_differences_d = bottom_differences_d[['b', 'd', 'b_score', 'd_score']]

            return bottom_differences_c, bottom_differences_d
        else:
            warnings.warn('Column d_score not found in the dataset')
            return super().find_bottom_differences(bottom_n)
        
if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv(f'/home/sperduti/phonetic_embeddings/results/CosineTest/TESTING_SCORE_COMPARATOR_D/comparator_d.csv')
    sc = ScoreComparator(data)
    sd = ScoreDifferenceFinder(data)
    scores = sc.compare_scores()
    top_differences = sd.find_top_differences(5)
    bottom_differences = sd.find_bottom_differences(5)
    with open(f'/home/sperduti/phonetic_embeddings/results/CosineTest/TESTING_SCORE_COMPARATOR_D/TEST_prevalences.txt', 'w') as f:
        f.write(f'homophone prevalence: {scores[0]}\n')
        f.write(f'Synonym prevalence: {scores[1]}\n')

    sc_4 = ScoreComparatorFour(data)
    sd_4 = ScoreDifferenceFinderFour(data)
    scores_4 = sc_4.compare_scores()
    top_differences_b_c, top_differences_b_d = sd_4.find_top_differences(5)
    bottom_differences_c, bottom_differences_d = sd_4.find_bottom_differences(5)
    with open(f'/home/sperduti/phonetic_embeddings/results/CosineTest/TESTING_SCORE_COMPARATOR_D/TEST_prevalences.txt', 'w') as f:
        f.write(f'homophone prevalence: {scores_4[0]}\n')
        f.write(f'Synonym prevalence: {scores_4[1]}\n')
        f.write(f'edit-distance prevalence: {scores_4[2]}\n')
        