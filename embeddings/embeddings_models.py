from abc import ABC, abstractmethod
from data.data_source import CMUdictionary2Vec
import numpy as np
import panphon
from transformers import AutoModel, AutoTokenizer

import gensim.downloader
import torch

import logging
logging.basicConfig(
    filename='log_problematic_Xphone.txt',
    level=logging.ERROR,  # Log only error-level messages
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def merge_dicts(data):
    # Check if the input is a dictionary
    if isinstance(data, dict):
        return data

    # Check if the input is a list of dictionaries
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        merged_dict = {}
        for d in data:
            merged_dict.update(d)
        return merged_dict

    else:
        raise ValueError("Input should be a dictionary or a list of dictionaries")

def check_same_keys(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    if keys1 != keys2:
        missing_keys1 = keys2 - keys1
        missing_keys2 = keys1 - keys2

        error_message = "Dictionaries not sharing the same keys! Check the dictionaries.\n"
        if missing_keys1:
            error_message += f"Keys missing in dict1: {missing_keys1}\n"
        if missing_keys2:
            error_message += f"Keys missing in dict2: {missing_keys2}"

        raise ValueError(error_message)


# A class that needs to be updated with the available models and that extracts the correct model's object given the right parameters.
class ModelFactory(ABC):

    @abstractmethod
    def create_model():
        pass

# Define an abstract base class for phonetic embeddings
class Embeddings(ABC):
    
    @abstractmethod
    def embed(self, sentence):
        pass

class Phoneme2Vec(Embeddings):
    def __init__(self, p2v_model, phonetic_dictionary = None) -> None:
        super().__init__()
        self.p2v_model = p2v_model
        self.keys = p2v_model.model.wv.key_to_index
        self.phonetic_dictionary = phonetic_dictionary

    def embed(self, sentence):
        if self.phonetic_dictionary is not None:
            embeddings = self.embed_dict(sentence)
        else:
            embeddings = self.embed_from_arp(sentence)
        return embeddings
    
    def embed_dict(self, sentence):
        sentence = sentence.lower().strip().split()
        phonetic_list = []
        for word in sentence: 
            word = [self.phonetic_dictionary.get('the') if self.phonetic_dictionary.get(word) is None else self.phonetic_dictionary.get(word)][0]
            phonetic_word = CMUdictionary2Vec.list_management_for_cmu(word)
            for phoneme in phonetic_word:
                index = self.keys[phoneme]
                phoneme_embedding = self.p2v_model[index]
                phonetic_list.append(phoneme_embedding)
        return np.mean(phonetic_list, axis=0)
    
    def embed_from_arp(self, sentence):
        
        def __preprocessing_sentence(sentence):
            import re
            if '   ' in sentence:
                n_sentence = []
                for element in sentence.split('   '):
                    element = re.sub(r'\s', '', element)
                    n_sentence.append(element)
                return n_sentence
            else:
                sentence.split()
                return sentence
        
        sentence = __preprocessing_sentence(sentence)
        phonetic_list = []
        sentence = sentence.split(' ')
        unknown_token = [0.0] * len(self.p2v_model[0])
        unk_count = 0
        for phoneme in sentence:
            if phoneme not in self.keys:
                if unk_count < 2:
                    unk_count += 1
                    continue
                else:
                    print(f'impossible to represent {sentence}. Vector with all 0 out.')
                    return unknown_token
            index = self.keys[phoneme]
            phoneme_embedding = self.p2v_model[index]
            phonetic_list.append(phoneme_embedding)
        return np.mean(phonetic_list, axis=0)

        
class ArticulatoryPhonemes(Embeddings):

    def __init__(self) -> None:
        super().__init__()
        self.ft = panphon.FeatureTable()

    def embed(self, input_phonemes):
        articulatory_characteristics = self.ft.word_fts(input_phonemes)
        embeddings = [art_char.numeric() for art_char in articulatory_characteristics]
        embeddings = np.mean(embeddings, axis=0)
        return embeddings

# Create a class for XPhoneBERT embeddings that inherits from phonetic_embeddings
class XPhoneBERT(Embeddings):

    def __init__(self) -> None:
        super().__init__()
        # Load the XPhoneBERT model and tokenizer
        self.model = AutoModel.from_pretrained("vinai/xphonebert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
        
    def embed(self, input_phonemes):
        # Tokenize the phonemes and obtain model features
        input_ids = self.tokenizer(input_phonemes, return_tensors="pt", truncation=True)  
        features = self.model(**input_ids)
        # Extract the embedding features from the model's output
        features = features.pooler_output
        return features

    def _find_consecutive(self, values, n):
        n = n-1
        values.sort()  
        for i in range(len(values) - n):
            if values[i] + n == values[i+n]:
                return values[i:i+(n+1)]
        return None

    def embed_from_sentence(self, input_phonemes, word, input_text_max_lenght = 512):
        # Tokenize the phonemes and obtain model features
        input_ids = self.tokenizer(input_phonemes, return_tensors="pt", truncation=True, max_length=input_text_max_lenght) 
        word_tokens = self.tokenizer.tokenize(word) 
        word_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(word_tokens))
        # torch.where: tensors must be broadcastable
        # input_ids_tensor.unsqueeze(0) making it of shape (1, input_ids_tensor_length).
        # word_ids_tensor.unsqueeze(1) making it of shape (word_ids_length, 1).
        # now they are broadcastable and can be used in torch.where
        word_ids = torch.where(input_ids.input_ids.unsqueeze(0) == word_ids.unsqueeze(1))
        # We sort the values; 
        word_ids = torch.sort(word_ids[2]).values
        # I want to extract embeddings only for the full word, which means that all the key phonemes should be consequential to each other
        word_ids = self._find_consecutive(word_ids, len(word_tokens))
        features = self.model(**input_ids)
        # Batch * sequence_length * hidden_size; : means "select all elements along this axis"
        features = features.last_hidden_state[:, word_ids,]
        # Average of the all the embeddings for all the phonemes
        features = torch.mean(features, dim=1)
        return features

class ClassicBERT(Embeddings):

    def __init__(self) -> None:
        super().__init__()
        # Load the classic BERT model and tokenizer
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def embed(self, input_text):
        # Tokenize the input text and obtain model features
        input_ids = self.tokenizer(input_text, return_tensors="pt")  
        features = self.model(**input_ids)
        
        # Check if majority of the tokens are known and not OOV
        known_tokens_mask = (input_ids["input_ids"] != self.tokenizer.unk_token_id) & (input_ids["input_ids"] != self.tokenizer.pad_token_id)

        if known_tokens_mask.sum() <= len(input_ids) // 2:
            # Extract the embedding features from the model's output using mean pooling
            features = torch.zeros_like(features.last_hidden_state[:, 0, :])
        else:
            features = features.last_hidden_state.mean(dim=1)
            # Set features to zeros if the majority of tokens are OOV
        return features
    
    def _find_consecutive(self, values, n):
        n = n-1
        values.sort()  
        for i in range(len(values) - n):
            if values[i] + n == values[i+n]:
                return values[i:i+(n+1)]
        return None

    # Extract the word embedding of a single word given a sentence (contextual embeddings)
    def embed_from_sentence(self, sentence, word, max_length = 512):
        # Tokenize the sentence and obtain model features
        input_ids = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length = max_length)  
        word_tokens = self.tokenizer.tokenize(word)
        word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
        if len(word_ids) == 1:
            word_ids = torch.where(input_ids["input_ids"] == word_ids[0])
            features = self.model(**input_ids)
            features = features.last_hidden_state[word_ids]
        else:
            word_ids = torch.tensor(word_ids)
            # torch.where: tensors must be broadcastable
            # input_ids_tensor.unsqueeze(0) making it of shape (1, input_ids_tensor_length).
            # word_ids_tensor.unsqueeze(1) making it of shape (word_ids_length, 1).
            # now they are broadcastable and can be used in torch.where
            word_ids = torch.where(input_ids.input_ids.unsqueeze(0) == word_ids.unsqueeze(1))
            # We sort the values;
            word_ids = torch.sort(word_ids[2]).values
            # I want to extract embeddings only for the full word, which means that all the key phonemes should be consequential to each other
            word_ids = self._find_consecutive(word_ids, len(word_tokens))
            features = self.model(**input_ids)
            features = features.last_hidden_state[:, word_ids,]
            # Average of the all the embeddings for all the phonemes
            features = torch.mean(features, dim=1)
        return features



# Purpose of this class: Select specific words in the context of a target word to extract embeddings.
# This is crucial when the context exceeds the length of the tokenizer, 
# so as to avoid losing key information when extracting contextual embeddings. E.g. It could happen that the 
# word we want to extract is after the truncation, and this should be avoided.
# Since XPhoneBert operates on phonemes, it also needs to select only relevant characters rather than phonemes.
class KeyContextExtractor:
    def __init__(self, model_max_length, tokenizer, window=10, windows_reduction_anyway=False):
        self.model_max_length = model_max_length
        self.window = window
        self.tokenizer = tokenizer
        self.windows_reduction_anyway = windows_reduction_anyway
        self.counter = 0

    def _check_sentence_tokens_length(self, sentence):
        tokenized_length = len(self.tokenizer(sentence)['input_ids'])
        if tokenized_length > self.model_max_length:
            return True
        return False
        
    def _solve_split_with_stemming(self, sentence, word):
        sentence_split = sentence.split(word)
        for i in range(1, len(word)):
            if len(sentence_split) >= 2:  # Make sure this is aligned properly with the 'for' loop
                break
            shortened_word = word[:-i]
            sentence_split = sentence.split(shortened_word)
        
        # If still no split, return the original word split
        if len(sentence_split) < 2:  # Ensure this 'if' block is aligned with the 'for' loop
            sentence_split = sentence.split(word)
        
        return sentence_split

    def check_sentence(self, sentence, word):
        are_tokens_more_than_max = self._check_sentence_tokens_length(sentence)
        if are_tokens_more_than_max or self.windows_reduction_anyway:  # Indentation looks good here
            word = word.lower()
            sentence = sentence.lower()
            sentence_split = self._solve_split_with_stemming(sentence, word)
            if len(sentence_split) < 2:
                self.counter += 1
                # Handle the case where the word is not found or sentence doesn't split into at least two parts
                if len(sentence_split) == 0:
                    sentence_split = ["", ""]
                    message = f"Word '{word}' not found in sentence: '{sentence}'."
                    print(message)
                    logging.error(message)
                elif len(sentence_split) == 1:
                    sentence_split = [sentence_split[0], ""]
                    message = f"Word '{word}' not found in sentence: '{sentence}'."
                    print(message)
                    logging.error(message)

            # Apply window reduction logic
            sentence = (
                self._window_reduction(sentence_split[0], 'l') + 
                [word] + 
                self._window_reduction(sentence_split[1], 'r')
            )
            sentence = ' '.join(sentence)
            print(f"Problematic cases so far: {self.counter}")
        
        return sentence    

    def _window_reduction(self, sentence, r_or_l):
        if r_or_l == 'l':
            sentence = sentence.split()
            if self.window > len(sentence):  # Ensure proper indentation of this block under 'if'
                return sentence
            sentence = sentence[-self.window:]
        elif r_or_l == 'r':
            sentence = sentence.split()
            if self.window > len(sentence):  # Ensure this block is indented correctly under 'elif'
                return sentence
            sentence = sentence[:self.window]
        return sentence

    

class Word2Vec(Embeddings):

    def __init__(self) -> None:
        super().__init__()
        self.tokens = gensim.downloader.load('word2vec-google-news-300')

    def embed(self, input_text):
        if input_text in self.tokens:
            return self.tokens[input_text]
        else:
            zero_analysis = [0.0] * len(self.tokens['the'])
            return zero_analysis

class CombinedModels():
    def __init__(self, model_1, model_2) -> None:
        self.model_1 = model_1
        self.model_2 = model_2
    
    def _istensor_check(embeddings):
        if isinstance(embeddings, torch.tensor):
            embeddings = embeddings.detach().numpy()
        return embeddings

    def embed(self, input_text):
        i_1 = self._istensor_check(self.model_1.embed(input_text))
        i_2 = self._istensor_check(self.model_2.embed(input_text))
        final_embs = np.stack[i_1, i_2]
        return final_embs
        
class CombinedModelsFromDict:
    def __init__(self, dict_1, dict_2) -> None:
        self.dict_1 = dict_1
        self.dict_2 = dict_2

    def _nan_check(self, array, array_comp):
        if isinstance(array, np.ndarray):
            return array
        if isinstance(array, list):
            return array
        if np.isnan(array):
            return np.zeros(len(array_comp))
        
    def _zeros_check(self, array):
        """
        Check if one of the arrays is a list full of zeros.

        Parameters:
        - array (numpy.ndarray): The first array to check.
        - array_comp (numpy.ndarray): The second array to compare with.

        Returns:
        - str: "EXCLUDED" if one of the arrays is a list full of zeros, otherwise, returns an empty string.
        """
        if isinstance(array, list):
            array = np.array(array)
        if np.all(array == 0):
            return "EXCLUDED"
        else:
            return array
        
    def combine_models(self):
        combined_embs = {}

        for element in self.dict_1.keys():
            array_1 = self.dict_1[element]
            array_2 = self.dict_2[element]
            # Check if the elements are PyTorch tensors
            if isinstance(self.dict_1[element], torch.Tensor):
                array_1 = self.dict_1[element].detach().numpy()[0]
            if isinstance(self.dict_2[element], torch.Tensor):
                array_2 = self.dict_2[element].detach().numpy()[0]
            
            array_1 = self._nan_check(array_1, array_2)
            array_2 = self._nan_check(array_2, array_1)

            array_1 = self._zeros_check(array_1)
            array_2 = self._zeros_check(array_2)

            if isinstance(array_1, str) or isinstance(array_2, str):
                combined_embs[element] = 'EXCLUDED'
                continue
            
            # Concatenate along the specified axis
            combined_embs[element] = np.array(list(array_1) + list(array_2))
            
        return combined_embs
    
class PhoneticModelFactory(ModelFactory):
    @staticmethod
    def create_model(model_type, p2v_model=None):
        """
        Factory class responsible for creating instances of phonetic models.

        :param model_type: Type of the phonetic model to create.
        :param p2v_model: Pre-trained model for Phoneme2Vec.
        :return: Instance of the specified phonetic model.
        """
        if model_type == 'XPhoneBERT':
            return XPhoneBERT()
        elif model_type == 'ArticulatoryPhonemes':
            return ArticulatoryPhonemes()
        elif model_type == 'Phoneme2Vec':
            return Phoneme2Vec(p2v_model)
        else:
            raise ValueError(f"Unsupported phonetic model type: {model_type}")
        
class SemanticModelFactory(ModelFactory):
    @staticmethod
    def create_model(model_type, w2v_model):
        """
        Factory class responsible for creating instances of phonetic models.

        :param model_type: Type of the phonetic model to create.
        :param p2v_model: Pre-trained model for Phoneme2Vec.
        :return: Instance of the specified phonetic model.
        """
        if model_type == 'ClassicBert':
            return ClassicBERT()
        elif model_type == 'Word2Vec':
            return Word2Vec()
        else:
            raise ValueError(f"Unsupported phonetic model type: {model_type}")
        
# Define the main function for testing
def main():
    # Sample sentences to be embedded

    from phon_utility.save_and_load import PickleLoader, PickleSaver

    loader = PickleLoader() 
    saver = PickleSaver()
    #BERT + P2V
    #BERT + articulatory
    #BERT + XPhone_BERT
    model = 'p2v'
    bert = merge_dicts(loader.load('/home/sperduti/phonetic_embeddings/embeddings/pretrained_embs/HomophoneTest/semantic_embs/bert_homophones.pkl'))
    phon = merge_dicts(loader.load('/home/sperduti/phonetic_embeddings/embeddings/pretrained_embs/HomophoneTest/phonetic_embs/' + model + '_homophones.pkl'))
    check_same_keys(bert, phon)
    #cmd = CombinedModelsFromDict(bert, phon)
    #cmd = cmd.combine_models()
    #path = '/home/sperduti/phonetic_embeddings/embeddings/pretrained_embs/HomophoneTest/mixed_embs/bert+' + model + '_homophones.pkl'
    #saver.save(cmd, path)
    
    embedded_word = 'are'
    test_sentence = 'hi how are you?'
    bert_embs = ClassicBERT()
    embedded_sentence = bert_embs.embed_from_sentence(test_sentence, embedded_word)


    print('hi')
    

if __name__ == '__main__':
    main()


