# transcription_handler.py
from data.data_source import TextToPhoneticDataset
from phon_utility.save_and_load import PrePhonDatasetLoader

# Each text (graphemes) must be converted either to IPA (the standardized phonetic alphabet) 
# or to ARPAbet (set of phonetic transcriptions) to extract phonetic based embeddings. 
# There are not models for ARPA automatic extraction to my knowledge, so you get the ARPA symbols of a model from
# CMU dictionary or other pre-processed sources (such as the PWE).
# This class will take some parameters and have the responsability of this extraction. 
class TranscriptionHandler:
    def __init__(self, training_set, grapheme_key, pretrained_phonemes = False) -> None:
        self.pretrained_phonemes = pretrained_phonemes
        self.training_set = training_set
        self.grapheme_key = grapheme_key
        self.grapheme_mapping = {}
        self.existing_transcription_loader = TextToPhoneticDataset(self.pretrained_phonemes) if self.pretrained_phonemes else None
        if self.existing_transcription_loader:
            # This passage wil create self.grapheme_mapping, which is a map of graphemes to phonemes
            self._load_existing_transcriptions(self.existing_transcription_loader)

    def _load_existing_transcriptions(self, dataset):
        pre_phon_load = PrePhonDatasetLoader(self.training_set, self.grapheme_key)
        pre_phon_load.load(dataset)
        self.grapheme_mapping = pre_phon_load.get_grapheme_mapping()

    # Responsability: it handles the trascription for two different types of dataset, list and DataDict. 
    # Can be extended also for other types in the future
    def _handle_transcription_batch(self, batch, text2phone_model, grapheme_phoneme=False):

        def _raise_value_error_if_the_not_in_grapheme_key():
            if not self.grapheme_mapping.get('the'):
                self.grapheme_mapping['the'] = 'DH AH'

        def _get_default_transcription(word, text2phone_model):
            _raise_value_error_if_the_not_in_grapheme_key()
            if word is None:
                word = 'the'
            if text2phone_model:
                if grapheme_phoneme:
                    return (word, self.grapheme_mapping.get(word) if self.pretrained_phonemes and self.grapheme_mapping.get(word) else text2phone_model.infer_sentence(word))
                return self.grapheme_mapping.get(word) if self.pretrained_phonemes and self.grapheme_mapping.get(word) else text2phone_model.infer_sentence(word)
            else:
                if grapheme_phoneme:
                    return (word, self.grapheme_mapping.get(word) if self.pretrained_phonemes and self.grapheme_mapping.get(word) else print('Not transcription found for word: ', word))
                return self.grapheme_mapping.get(word) if self.pretrained_phonemes and self.grapheme_mapping.get(word) else 'the'

        if isinstance(batch, list):
            batch_dict = {}
            # This is the case of loading ARPAbet transcriptions. Not all the words exist in ARPAbet, so a new type of handling
            # is necessary. In case of ARPA, Text2phone model will not be present and we will load a dictionary of word:ARPAtranscription
            # In case of non-existing words, we are selecting "the", which is the most common word in English, which will work as a baseline.
            batch_dict['transcription'] = [_get_default_transcription(word, text2phone_model) for word in batch]
            batch = batch_dict
        else:
            batch[self.grapheme_key] = [b if b is not None else 'the' for b in batch[self.grapheme_key]]
            batch['transcription'] = [_get_default_transcription(word, text2phone_model) for word in batch[self.grapheme_key]]  
        return batch  

class TranscriptionEasy:
    def __init__(self, training_set) -> None:
        self.grapheme_mapping = training_set

    def _get_default_transcription(self, word):
        return self.grapheme_mapping.get(word) if self.grapheme_mapping.get(word) else 'the'

    def _handle_transcription_batch(self, batch):
        """
        Processes a batch of words and returns a dictionary in the required format.

        Args:
            batch: A list of words.

        Returns:
            A dictionary with the key 'transcription' and the values being word-transcription pairs.
        """

        # Create a dictionary to store the processed words and their transcriptions
        word_transcription_list = []

        for word in batch:
            # Get the default transcription for the word
            transcription = self._get_default_transcription(word)

            # Add the word-transcription pair to the dictionary
            word_transcription_tuple = (word, transcription)
            word_transcription_list.append(word_transcription_tuple)

        return word_transcription_list