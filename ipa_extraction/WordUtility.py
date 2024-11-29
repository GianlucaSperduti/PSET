""" Module: contaning utility functions for word processing, such as extracting unique words from a dataset. """

from abc import ABC, abstractmethod
import string


from datasets import dataset_dict

class ExtractorInterface(ABC):
    def __init__(self) -> None:

        @staticmethod
        @abstractmethod
        def extract_unique_words(dataset: str) -> list:
            """ Extracts unique words from a dataset and returns a list of unique words. """
            pass

class Writer(ABC):
    def __init__(self) -> None:

        @staticmethod
        @abstractmethod
        def write_unique_words(unique_words: list) -> None:
            """ Prints the unique words to the console. """
            pass

class UniqueWordsExtractor(ExtractorInterface):

    def extract_unique_words(self, dataset):
        if isinstance(dataset, str) and dataset.endswith('.txt'):
            return self._extract_unique_words_txt(dataset)
        elif isinstance(dataset, list):
            return self._extract_unique_words_list(dataset)
        else:
            raise ValueError('Dataset format not supported.')

    @staticmethod
    def _extract_unique_words_txt(dataset_path: str) -> list:
        """ Extracts unique words from a dataset and returns a list of unique words. """
        with open(dataset_path, 'r') as dataset_file:
            all_lines = dataset_file.readlines()
            words = [line.split('\t')[0] for line in all_lines]
            unique_words = list(set(words))
            return unique_words

    @staticmethod
    def _extract_unique_words_list(dataset: list) -> list:
        """ Extracts unique words from a dataset and returns a list of unique words. """
        words_list = [words.split() for words in dataset]
        words_list = [word for sublist in words_list for word in sublist]
        unique_words = set(words_list)
        return unique_words

class UniqueWordsCleanAndExtract(UniqueWordsExtractor):
    def __init__(self) -> None:
        super().__init__()

    """ Class used to extract unique words from a dataset. It removes the punctuation in advance. """

    @staticmethod
    def _extract_unique_words_txt(dataset_path: str) -> list:
        """ Extracts unique words from a dataset and returns a list of unique words. """
        with open(dataset_path, 'r') as dataset_file:
            all_lines = dataset_file.readlines()
            words_list = [line.split('\t')[0] for line in all_lines] 
            words_list = [UniqueWordsCleanAndExtract._clean_text(word) for word in words_list]
            unique_words = list(set(words_list))
            return unique_words

    @staticmethod
    def _extract_unique_words_list(dataset: list) -> list:
        """ Extracts unique words from a dataset and returns a list of unique words. """
        words_list = [words.split() for words in dataset]
        words_list = [word for sublist in words_list for word in sublist]
        words_list = [UniqueWordsCleanAndExtract._clean_text(word) for word in words_list]
        unique_words = set(words_list)
        return unique_words

    @staticmethod
    def _remove_punctuation(text):
        # Create a translation table to map punctuation characters to None
        translator = str.maketrans('', '', string.punctuation)
        # Use translate method to remove punctuation characters
        return text.translate(translator)
    
    @staticmethod
    def _remove_quotes(text):
        return text.replace("'", "")

    @staticmethod
    def _remove_double_quotes(text):
        return text.replace('"', '')    

    @staticmethod
    def _remove_capitalization(text):
        return text.lower()

    staticmethod
    def _clean_text(text):
        text = UniqueWordsCleanAndExtract._remove_punctuation(text)
        text = UniqueWordsCleanAndExtract._remove_quotes(text)
        text = UniqueWordsCleanAndExtract._remove_double_quotes(text)
        text = UniqueWordsCleanAndExtract._remove_capitalization(text)
        return text

class WordsWriter(Writer):

    @staticmethod
    def write_to_file(words, filename):
        with open(filename, 'w') as file:
            for word in words:
                file.write(word + '\n')
