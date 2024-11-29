# batch_processing.py
from abc import ABC, abstractmethod

from phon_utility.save_and_load import BatchSaver
import numpy as np

class BatchExtractor(ABC):
    def __init__(self) -> None:
        super().__init__()

        @abstractmethod
        def _extract_batch_embeddings():
            pass

        @abstractmethod
        def _extract_batch_embeddings_as_dict():
            pass

class TrainingBatch:
    def __init__(self) -> None:
        self.batch = []

    # Responsability: it gets the batch in the right format 
    def _get_batch_gk(self, batch_size, i, training_set, grapheme_key):
        self.batch = [training_set[grapheme_key][i:i + batch_size]
                        if isinstance(training_set, dict)
                        else training_set[i:i + batch_size]][0]
    
    # Responsability: it gets the batch in the right format 
    def _get_batch(self, batch_size, i, training_set):
        """
        Gets a batch of data from the training set in the desired format.

        Args:
            batch_size (int): Size of the batch to retrieve.
            i (int): Starting index of the batch.
            training_set: The training data (can be a dictionary or another iterable).

        Returns:
            list: A list containing the requested batch of data.
        """

        # Check if training_set is a dictionary
        if isinstance(training_set, dict):
            # Create a temporary list copy to avoid modifying the original dictionary
            temp_list_k = list(training_set.keys())
            # Extract the batch using slicing
            self.batch = temp_list_k[i:i + batch_size]
        else:
            # If not a dictionary, use regular slicing
            self.batch = training_set[i:i + batch_size]

        return self.batch

        
class BatchPhoneticEmbsExtractor(BatchExtractor):
    def __init__(self, phonetic_model, output_file, batch_size):
        self.phonetic_model = phonetic_model
        self.extracted_embeddings = []
        self.output_file = output_file
        self.batch_size = batch_size

    # Responsability: after the phonetic transcript extraction, it eventually handles embeddings extractin, batch per batch.
    def _extract_batch_embeddings(self, i, batch_size, batch):
        for item in batch:
            sentence = item if item is not None else 'ə'
            embeddings = self.phonetic_model.embed(sentence)
            self.extracted_embeddings.append(embeddings)

        if i % batch_size == 0:
            BatchSaver().save_batch(self.extracted_embeddings, self.output_file, i // self.batch_size)
            self.extracted_embeddings = []

    def check_if_multi_words(self, item):
        if len(item.split()) > 1:
            return True
        else:
            return False
        
    def check_if_contextual_model(self):
        if hasattr(self.phonetic_model, 'embed_from_sentence'):
            return True
        else:
            return False  
    
    def model_embed_multi_words(self, item):
        embs = []
        for element in item.split():
            embeddings = self.phonetic_model.embed(element)
            embs.append(embeddings)
        embs = [emb for emb in embs if str(emb) != 'nan']
        embs = sum(embs) / len(embs)
        return embs

    def _extract_batch_embeddings_as_dict(self, i, batch_size, batch):
        # If it's a contextual model, we will extract the "sentence" embeddings"
        is_a_contextual_model = self.check_if_contextual_model()
        embs_dict = {}
        for item in batch:
            sentence = item[1] if item[1] is not None else 'ə'
            if self.check_if_multi_words(sentence) and is_a_contextual_model == False:
                embeddings = self.model_embed_multi_words(sentence)
                embs_dict[item[0]] = embeddings
            else:
                embeddings = self.phonetic_model.embed(sentence)
                embs_dict[item[0]] = embeddings

        self.extracted_embeddings.append(embs_dict)

        if i % batch_size == 0:
            BatchSaver().save_batch(self.extracted_embeddings, self.output_file, i // self.batch_size)
            embs_dict = {}
            self.extracted_embeddings = []  

class BatchEmbsExtractor(BatchExtractor):
    def __init__(self, model, output_file, batch_size):
        self.model = model
        self.extracted_embeddings = []
        self.output_file = output_file
        self.batch_size = batch_size

    # Responsability: after the phonetic transcript extraction, it eventually handles embeddings extractin, batch per batch.
    def _extract_batch_embeddings(self, i, batch_size, batch):
        for item in batch:
            sentence = item if item is not None else 'ə'
            embeddings = self.model.embed(sentence)
            self.extracted_embeddings.append(embeddings)

        if i % batch_size == 0:
            BatchSaver().save_batch(self.extracted_embeddings, self.output_file, i // self.batch_size)
            self.extracted_embeddings = []  

    def check_if_multi_words(self, item):
        if len(item.split()) > 1:
            return True
        else:
            return False
        
    def check_if_contextual_model(self):
        if hasattr(self.model, 'embed_from_sentence'):
            return True
        else:
            return False

    def model_embed_multi_words(self, item):
        embs = []
        for element in item.split():
            embeddings = self.model.embed(element)
            embs.append(embeddings)
        embs = sum(embs) / len(embs)
        return embs
        
    def _extract_batch_embeddings_as_dict(self, i, batch_size, batch):
        # If it's a contextual model, we will extract the "sentence" embeddings"
        is_a_contextual_model = self.check_if_contextual_model()
        embs_dict = {}
        for item in batch:
            sentence = item[1] if item[1] is not None and str(item[1]) != 'nan' else 'ə'
            if self.check_if_multi_words(sentence) and is_a_contextual_model == False:
                embeddings = self.model_embed_multi_words(sentence)
                embs_dict[item[0]] = embeddings
            else:
                embeddings = self.model.embed(sentence)
                embs_dict[item[0]] = embeddings

        self.extracted_embeddings.append(embs_dict)

        if i % batch_size == 0:
            BatchSaver().save_batch(self.extracted_embeddings, self.output_file, i // self.batch_size)
            embs_dict = {}
            self.extracted_embeddings = []  



