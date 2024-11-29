from numpy import single
import pandas as pd
from phon_utility.save_and_load import PickleLoader
import os

datset_homophone_path = '../phonetic_embeddings/data/datasets/SemPhonDiscr/Homophone_GOLD_STANDARD.csv'
a_h_s = 'homophones'
pkl_path = os.path.join('../phonetic_embeddings/data/datasets/SemPhonDiscr/sentences/classic', a_h_s + '.pkl')

pl = PickleLoader()
dict_of_words = pl.load(pkl_path)
single_words = [word for word, _ in dict_of_words.items()]
len_of_new_single_words = len(single_words)
set_of_new_single_words = set(single_words)

dataframe = pd.read_csv(datset_homophone_path)
len_of_original_datasframe = len(dataframe[a_h_s])
set_of_original_datasframe = set(dataframe[a_h_s])

missing_words = set_of_new_single_words.difference(set_of_original_datasframe)

print(f"Original len: {len_of_original_datasframe}")
print(f"New len: {len_of_new_single_words}")
print(f"Missing words: {missing_words}")

for x in list(dataframe[a_h_s]):
    if list(dataframe[a_h_s]).count(x) > 1:
        print(x)
        print(list(dataframe[a_h_s]).count(x))
                