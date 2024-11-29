from typing import Any
from abc import ABC, abstractmethod
import csv

import numpy as np
import os
from sklearn.decomposition import PCA
import torch

def string_to_bool(input_str):
    lower_str = input_str.lower()
    if lower_str == "true":
        return True
    elif lower_str == "false":
        return False
    else:
        raise ValueError(f"Invalid input: {input_str}")

def string_to_bool_rt(input_str):
    lower_str = input_str.lower()
    if lower_str == "true":
        return True
    elif lower_str == "false":
        return False
    else:
        return input_str

class checker(ABC):
    def __init__(self) -> None:
         super().__init__()

    def check(self):
         pass
    
class CsvProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def process_csv():
        pass
    
class DictionaryProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def process_dict(self, dictionary):
        pass


class dimensionality_reducer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reduce_dimension(matrix, n_components):
        pass

class format_handler_interface(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def format_data(self, object):
        pass

class FolderOpeartion(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def operate(self):
        pass     

class matrix_handler(format_handler_interface):
    def __init__(self) -> None:
        super().__init__()

    def detach_tensor(self, tensor):
            tensor = tensor.detach().numpy()
            return tensor
    
    def format_data(self, matrix):
            if isinstance(matrix[0], torch.Tensor):
                matrix = [self.detach_tensor(tensor) for tensor in matrix]
                matrix = np.squeeze(matrix, axis=1)
            return matrix
    
class TrainingDataBatcher(format_handler_interface):
    def __init__(self) -> None:
        super().__init__()
    
    def format_data(self, dataset, grapheme_key):
        self.training_set = [dataset['train'] if isinstance(dataset, dict) else dataset.dataset['train']][0]
        self.length = [len(dataset['train'][grapheme_key]) if isinstance(dataset, dict) else len(dataset.dataset['train'])][0]


class LoadSpecDeterminer(format_handler_interface):
    def __init__(self, arp_key, transcription_models=['XPhoneBERT', 'ArticulatoryPhonemes']):
        self.arp_key = arp_key
        self.transcription_models = transcription_models

    def format_data(self, phonetic_model):
        # If more models are inserted 
        for t_m in self.transcription_models:
            if t_m in str(phonetic_model):
                return 'transcription'
        else:
            return self.arp_key

class PCA_reducer(dimensionality_reducer):
    def __init__(self, n_components) -> None:
        super().__init__()
        self.n_components = n_components

    def reduce_dimension(self, matrix):
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(matrix)
        return X_pca

class PathChecker(checker):
    def __init__(self) -> None:
            super().__init__()

    def check(self, path):
        if os.path.exists(path):
            existance = True
        else:
            existance = False
        return existance
    
class HighestNumberInFolder(FolderOpeartion):
    def __init__(self) -> None:
         super().__init__()
    
    def operate(self, folder_path):
        # Get a list of all files in the folder
        files = os.listdir(folder_path)

        # Filter out directories from the list
        files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

        # If there are no files, return None
        if not files:
            return None

        # Extract numeric parts from the filenames
        file_numbers = [int(''.join(filter(str.isdigit, f))) for f in files]

        # Find the file with the highest numeric part
        highest_numbered_file_index = max(file_numbers)

        return highest_numbered_file_index

class WLCsvProcessor(CsvProcessor):
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def process_csv(self):
        # Read the input CSV file
        with open(self.input_file, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # Assuming the first row is the header
            rows_to_include = []

            # Iterate through the rows and identify rows to be excluded
            for row in csv_reader:
                if row and len(row) > header.index('TO_BE_EXCLUDED') and row[header.index('TO_BE_EXCLUDED')] == 'TO_BE_EXCLUDED':
                    pass
                else:
                    rows_to_include.append(row)

        # Write the output CSV file excluding rows to be excluded and last 5 columns
        with open(self.output_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write the header excluding the last 5 columns
            csv_writer.writerow(header[:-5])

            with_excluded = 0  # Counter for rows with TO_BE_EXCLUDE set to True
            for row in rows_to_include:
                # Write the row excluding the last 5 columns
                csv_writer.writerow(row[:-5])
                with_excluded += 1

            print(f"{with_excluded} rows included set to True excluded.")


class DictValueSorter(DictionaryProcessor):
    # The process_dict method sorts the elements of a dictionary based on the values associated with each key.
    # Parameters:
    #   - dictionary: The input dictionary to be processed.
    #   - reverse: If True, sorts the dictionary in descending order (default is True).
    #   - k: Specifies the number of top elements to retain after sorting (default is 10).
    # Returns:
    #   - A dictionary containing the top k elements for each key, sorted by their values.
    # Note: This method does not modify the original dictionary but returns a new sorted dictionary.
    # e.g. 
    # k = 3; reverse = True;
    # input_dict = input_dictionary = {
    #                                'key1': {'a': 5, 'b': 3, 'c': 8, 'd': 1},
    #                                'key2': {'x': 10, 'y': 6, 'z': 2},
    #                                'key3': {'foo': 7, 'bar': 4, 'baz': 9}
    #                            }
    # e.g. output = output_dictionary = 'key1': {'c': 8, 'a': 5, 'b': 3, 'd': 1},
    #                                'key2': {'x': 10, 'y': 6, 'z': 2},
    #                                'key3': {'baz': 9, 'foo': 7, 'bar': 4}
    @staticmethod
    def process_dict(dictionary, reverse=True, k=10):
        sorted_elements = {}
        for key, item in dictionary.items():
            sorted_dict = sorted(dictionary[key].items(), key=lambda item: item[1], reverse=reverse)[:k]
            sorted_elements[key] = sorted_dict
        return sorted_elements

class FromDictToCsv(DictionaryProcessor):
    @staticmethod
    def process_dict(dictionary):
        only_words_csv_dict = {}
        for key, _ in dictionary.items():
            only_words_csv_dict[key] = [item[0] for item in dictionary[key]]
        return only_words_csv_dict




if __name__ == "__main__":
    input_file_path = "/home/sperduti/phonetic_embeddings/wfe/spelling_variants.csv"  # Replace with the actual file path
    output_file_path = "word_list_human_ev.txt"  # Replace with the desired output file path

    processor = WLCsvProcessor(input_file_path, output_file_path)
    processor.process_csv()