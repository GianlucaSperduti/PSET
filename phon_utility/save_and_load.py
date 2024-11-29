from abc import ABC, abstractmethod
import pickle
import os
import pandas as pd
from typing import Dict

class ModelLoader(ABC):
    @staticmethod
    @abstractmethod
    def load(pkl_obj):
        pass

class ModelSaver(ABC):    
    @staticmethod          
    @abstractmethod
    def save(pkl_obj, save_path, return_path, mode):
        pass

class PickleSaver(ModelSaver):
    @staticmethod
    def save(pkl_obj, save_path, return_path = False, mode = 'wb'):
        with open(save_path, mode) as file:
            pickle.dump(pkl_obj, file)
            if return_path == True:
                return save_path 
            
class BatchSaver(PickleSaver):
    def save_batch(self, data, save_path, batch_index):
        batch_filename = f"{save_path}/batch_{batch_index}.pkl"
        super().save(data, batch_filename, mode='wb')

class BatchConcatenator:
    def concatenate_batches(self, save_path, batch_path, num_batches):
        final_embeddings = []  # Initialize an empty list to hold the concatenated embeddings

        for i in range(num_batches+1):
            batch_filename = f"{batch_path}/batch_{i}.pkl"
            if os.path.exists(batch_filename):
                with open(batch_filename, 'rb') as batch_file:
                    # Load the pickled data from the batch file
                    embeddings_batch = pickle.load(batch_file)

                    # Extend the final_embeddings list with the embeddings from the current batch
                    final_embeddings.extend(embeddings_batch)
                  
                # Remove the processed batch file (optional)
                os.remove(batch_filename)

        final_dict = {}
        for dictionary in final_embeddings:
            final_dict.update(dictionary)

        # Save the concatenated embeddings to the final file
        with open(save_path, 'wb') as final_file:
            pickle.dump(final_dict, final_file)

class PickleLoader(ModelLoader):
    @staticmethod
    def load(pkl_obj):
        with open(pkl_obj, 'rb') as file:
            return pickle.load(file)

class BatchLoader(ModelLoader):
    @staticmethod
    def load(save_path, batch_index):
        batch_filename = f"{save_path}/batch_{batch_index}.pkl"
        with open(batch_filename, 'rb') as file:
            return pickle.load(file) 
        
class PrePhonDatasetLoader(ModelLoader):
    def __init__(self, training_set, grapheme_key):
        self.training_set = training_set
        self.grapheme_key = grapheme_key
        self.grapheme_mapping = {}

    def load(self, loader):
        data_dict = loader.get_data()
        self.grapheme_mapping = {b: data_dict.get(b) for b in self.training_set[self.grapheme_key]}

    def get_grapheme_mapping(self) -> Dict:
        return self.grapheme_mapping


# Function to load and quadruplicate the dataset
def load_and_quadruplicate_dataset(filepath):
    original_dataset = pd.read_csv(filepath)
    original_dataset = original_dataset.loc[original_dataset.index.repeat(4)].reset_index(drop=True)
    original_dataset['ID'] = original_dataset.index // 4 + 1  # ID starts from 1
    original_dataset['answers'] = ''
    return original_dataset

def load_4_prompts(filepath):
    original_dataset = pd.read_csv(filepath)
    return original_dataset, original_dataset, original_dataset, original_dataset