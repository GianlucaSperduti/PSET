""" Module containing the IpaExtractor class, which is used to extract IPA transcriptions from a dataset. """

from abc import ABC, abstractmethod

from text2phonemesequence import Text2PhonemeSequence
from tqdm import tqdm
import os
from os.path import join 

class Transcriptor(ABC):
    """
    Abstract class for transcription extraction.
    """
    @abstractmethod
    def generate_phonetic_transcriptions(self, path_dict, output_path):
        pass

class IpaTranscription(Transcriptor):
    """
    Class used to extract IPA transcriptions from a dataset.
    """
    def __init__(self) -> None:
        """
        Initializes the IpaExtractor object. """

    @staticmethod
    def generate_phonetic_transcriptions(input_path, output_path, batch_size=64, language='en-us'):
        """
        Generates phonetic transcriptions for a given input file.

        Args:
            input_path (str): The path to the input file.
            output_path (str): The path to save the output file.
            batch_size (int, optional): The batch size for processing the input file. Defaults to 64.
            language (str, optional): The language for generating the phonetic transcriptions. Defaults to 'en-us'.

        Returns:
            None
        """
        t2p = Text2PhonemeSequence(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_tiny_16_layers_100', language=language, is_cuda=True)
        t2p.infer_dataset(input_file=input_path, output_file=output_path, batch_size=batch_size)

        print(f"Phonetic transcriptions generated. Output file: {output_path}")

class IpaTranscriptionSentence(Transcriptor):
    """
    Class used to extract IPA transcriptions from a dataset.
    """
    def __init__(self) -> None:
        """
        Initializes the IpaExtractor object. """

    @staticmethod
    def generate_phonetic_transcriptions(sentence_list, language='en-us'):
        """
        Generates phonetic transcriptions using Text2PhonemeSequence.

        Parameters:
        - path_dict (List[dict]): List of dictionaries with path and lang entries. 
        - output_path (str): The path to the output folder.

        Returns:
        - None
        """
        t2p_l = []
        t2p = Text2PhonemeSequence(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_tiny_16_layers_100', language=language, is_cuda=True)
        for s in tqdm(sentence_list):
            trans_t2p = t2p.infer_sentence(s)
            t2p_l.append(trans_t2p)
        return t2p_l
        