from abc import ABC, abstractmethod
from typing import List
from datasets import load_dataset
from nltk.corpus import cmudict
import nltk
import os
import pandas as pd
import random
from typing import Dict
from phon_utility.save_and_load import PickleLoader

class DatasetUtility(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply():
        pass

class DFInterface(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def create_dataset():
        pass

class dataset(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.dataset = []

class CMUdictionary(dataset):
    def __init__(self):
        super().__init__()
        nltk.download('cmudict')
        self.dataset = cmudict.dict()

    def __repr__(self) -> str:
        repr = f'CMU Dictionary. Elements in the dataset: {len(self.dataset)}'
        return repr

    def __getitem__(self, index):
        return self.dataset[index]

class CMUdictionary2Vec(CMUdictionary):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = CMUdictionary2Vec.preparing_for_p2v(self.dataset)

    # this static function should probably be out of this particular class. 
    # this handles the CMU dictionary particularities when extracting phonemese.
    # cases (made-up examples, just to explain):
    # like a -> [[ae][ei]]; hello -> [['h','e','l','o']; ['h','e','l','ou']], 'hi' -> [[ai]]
    @staticmethod
    def list_management_for_cmu(phonemes_list):
        if len(phonemes_list)>1:
            for phoneme in phonemes_list: 
                if len(phoneme) > 1: 
                    phonemes_list = phonemes_list[0]
                    return phonemes_list
                else:
                    phonemes_list = [phon[0] for phon in phonemes_list]
                    return phonemes_list
        else: 
            phonemes_list = phonemes_list[0]
        return phonemes_list
    
    def preparing_for_p2v(self):
            phoneme_sequences = []
            for word, phonemes_list in self.items():
                # Use the first pronunciation variant for simplicity
                phonemes_list = CMUdictionary2Vec.list_management_for_cmu(phonemes_list)
                phoneme_sequences.append(phonemes_list)
            phonetic_dataset = phoneme_sequences
            return phonetic_dataset

class PWEsuite(dataset):
    def __init__(self) -> None:
        super().__init__() 
        self.dataset = load_dataset("zouharvi/pwesuite-eval")

class PWEsuite4p2v(PWEsuite):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = [token_arp.split() for token_arp in self.dataset['train']['token_arp']]


class PWEsuite4p2v2(PWEsuite4p2v):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = [token_arp.split() for token_arp in self.dataset['train']['token_arp']]
        for element in self.dataset:
            print(element)
            break

class TextToPhoneticDataset(dataset):
    def __init__(self, file_path, secondary_file_path='', train_format=False):
        self.file_path = file_path
        # This is useful in case you have used the IPA/ARPABET extraction algorithm to obtain the translated dataset 
        # and want to recompose it. It is only implemented for the csv options by now (following the extractoin algorithms)
        self.secondary_file_path = secondary_file_path
        # Load data based on file extension
        if ".pkl" in self.file_path:
            self.dataset = PickleLoader().load(file_path)
        elif '.txt' in self.file_path:
            self.dataset = self.load_data_txt()
        elif '.csv' in self.file_path:
            self.dataset = self.load_data_csv()
        else:
            raise ValueError(f"Format {self.file_path.split('.')[1]} not supported by PrePhonDataset class.")
        
        if train_format:
            self.dataset = self.dataset

    def load_data_csv(self):
        """ 
        Loads data from a csv file.

        Returns:
            dict: Dictionary containing the loaded data.
        """

        # This is the simple case 
        dataset = pd.read_csv(self.file_path)
        dataset = list(dataset.values.flatten())

        if self.secondary_file_path != '':
            secondary_dataset = pd.read_csv(self.secondary_file_path)
            secondary_dataset = list(secondary_dataset.values.flatten())
            dataset = [{a:b} for a, b in zip(dataset, secondary_dataset)]

            # Initialize an empty list to store the flattened content
            flattened_dict = {}

            # Iterate over each dictionary in the list
            for entry in dataset:
                # Concatenate keys and values into the flattened list
                flattened_dict.update(entry)

        dataset = flattened_dict
        return dataset

    def load_data_txt(self):
        """
        Loads data from a text file.

        Returns:
            dict: Dictionary containing the loaded data.
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            dataset = {}  # Initialize empty dictionary

            for line in lines:
                parts = line.strip().split('|')

                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    dataset[key] = value  # Add data to the dictionary
                
                if len(parts) == 1:
                    key = parts[0].strip()
                    dataset[key] = key

        return dataset

    def get_data(self):
        """
        Returns the loaded data.

        Returns:
            dict: The loaded data.
        """
        return self.dataset

class SUBTLEX_UK(dataset):
    def __init__(self, path_to_SUBTLEX_UK='dataset/SUBTLEX-UK.xlsx') -> None:
        self.dataset = pd.read_excel(path_to_SUBTLEX_UK)

    def __getitem__(self, index):
        return self.dataset.iloc[index]

class SUBTLEX_UK_filtered(SUBTLEX_UK):
    def __init__(self, path_to_SUBTLEX_UK='dataset/SUBTLEX-UK.xlsx', zipf_limit=5) -> None:
        super().__init__(path_to_SUBTLEX_UK)  # Fix the syntax here
        self.dataset = self.dataset[['Spelling', 'LogFreq(Zipf)']][self.dataset['LogFreq(Zipf)'] > zipf_limit].sort_values(by='LogFreq(Zipf)', ascending=False)

class SemanticPhoneticTest(dataset):

    def __init__(self, path_to_dataset: str, anch: List[str], b: List[str], synonym_dataset: pd.DataFrame, save: bool = True, synonyms: bool = False) -> None:
        super().__init__()
        self.anch = anch
        self.b = b
        self.c = [self._random_synonyms_extraction(self.anch, synonym_dataset) if not synonyms else synonyms][0]
        self.dataset = self._create_dataset(path_to_dataset, save)
        self._remove_empty_synonyms()
        
    def _random_synonyms_extraction(anchors: List[str], synonym_dataset: pd.DataFrame):
        synonyms = list()
        for w in anchors:
            synonym_set = synonym_dataset['c'][synonym_dataset['word']==w]
            if len(synonym_set) == 0:
                synonyms.append('')
                continue
            random_synonym = synonym_set.iloc[random.randint(0, (len(synonym_dataset['c'][synonym_dataset['word']==w]))-1)]
            if len(random_synonym) == 0:
                synonyms.append('')
                continue
            synonyms.append(random_synonym[random.randint(0, (len(random_synonym)-1))])
        return synonyms

    def _create_dataset(self, path_to_dataset: str, save: bool) -> pd.DataFrame:
        if os.path.exists(path_to_dataset):
            return pd.read_csv(path_to_dataset, index_col=False)
        else:
            columns = ['a', 'b', 'c']
            df = pd.DataFrame(columns=columns)
            df['a'] = self.a
            df['b'] = self.b
            df['c'] = self.c
            if save:
                df.to_csv(path_to_dataset)

            # Drop rows where any column has a value of None
            df = df.dropna()

            return df
        
    def _remove_empty_synonyms(self):
        self.dataset = self.dataset[self.dataset['synonyms'] != '']
        self.dataset = self.dataset.dropna(subset=['synonyms'])


class CosineDatasetUtility(DatasetUtility):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def apply(dataset):
        tuples_of_words = []
        for l in range(len(dataset)):
            instance = dataset.iloc[l]
            instance = tuple(instance)
            tuples_of_words.append(instance)       
        return tuples_of_words

class DatasetFactory(DFInterface):
    def __init__(self, dataset, file_path, phonetic_path='') -> None:
        self.dataset = dataset
        self.file_path = file_path
        self.phonetic_path = phonetic_path

    def create_dataset(self):
        if self.dataset == 'PWEsuite':
            return PWEsuite()
        elif self.dataset == 'PhoneticDataset' or self.dataset == 'SemanticDataset':
            return TextToPhoneticDataset(self.file_path, self.phonetic_path, train_format=True)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset}. Add datasets through data_source.DatasetFactory._create_dataset_instance")



def main():
    # Example usage:
        from phon_utility.save_and_load import PickleLoader

        pl = PickleLoader()
        
        anchors = list(pl.load('/home/sperduti/phonetic_embeddings/data/datasets/SemPhonDiscr/Homophone_Synonyms/anchors.pkl'))
        homophones = list(pl.load('/home/sperduti/phonetic_embeddings/data/datasets/SemPhonDiscr/Homophone_Synonyms/homophones.pkl'))
        synonyms = pd.read_json('/home/sperduti/phonetic_embeddings/thesaurus/en_thesaurus.jsonl', lines=True)


if __name__ == '__main__':
    main()
