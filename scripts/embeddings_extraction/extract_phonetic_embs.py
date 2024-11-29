import argparse
import json
import time
from tqdm import tqdm
from data.data_source import DatasetFactory
from embeddings.embeddings_models import PhoneticModelFactory
from phon_utility.save_and_load import PickleLoader, BatchConcatenator
from text2phonemesequence import Text2PhonemeSequence
from phon_utility.phon_utility import HighestNumberInFolder 
from SemPhonTest.TranscriptionHandler import TranscriptionEasy
from SemPhonTest.BatchProcessing import TrainingBatch, BatchPhoneticEmbsExtractor

class PhoneticEmbeddingsProcessor:
    def __init__(self, args): 
        self._load_configuration(args)
        self._initialize_components()

    def _load_configuration(self, args):
        self.dataset_config = args.dataset
        self.file_path = args.file_path
        self.phonetic_dataset = args.secondary_phonetic_path
        self.dataset_config_selector = DatasetFactory(self.dataset_config, self.file_path, self.phonetic_dataset)
        self.dataset = self.dataset_config_selector.create_dataset()
        self.training_set, self.length = self.dataset.dataset, len(self.dataset.dataset)
        self.embeddings_path = args.embeddings_path
        self.load_last_batch_bool = args.load_last_batch.lower() == 'true'
        self.last_batch = self._calculate_last_batch() if self.load_last_batch_bool else 0
        self.p2v_model = [PickleLoader().load(args.p2v_model) if args.p2v_model != '' else ''][0]
        self.phonetic_model = PhoneticModelFactory.create_model(args.phonetic_model, self.p2v_model)  
        self.text2phone_model = self._initialize_text2phone_model()
        self.batch_size = int(args.batch_size)

    def _calculate_last_batch(self):
        batch_calculator = HighestNumberInFolder()
        last_batch = batch_calculator.operate(self.embeddings_path)
        if last_batch is None:
            last_batch = 0
            return last_batch
        return last_batch - 1
    
    def _initialize_components(self):
        self.training_batch = TrainingBatch()
        self.batch_processor = BatchPhoneticEmbsExtractor(self.phonetic_model, self.embeddings_path, self.batch_size)
        self.last_batch = self._calculate_last_batch() if self.load_last_batch_bool else 0
        self.transcription_handler = TranscriptionEasy(self.training_set)
        return self.training_batch

    def _initialize_text2phone_model(self):
        return Text2PhonemeSequence(language='en-us', is_cuda=True)

    def extract_embeddings(self):
        for i in tqdm(range(self.last_batch * self.batch_size, self.length, self.batch_size), desc="Processing batches"):
            self.training_batch._get_batch(self.batch_size, i, self.training_set)
            self.training_batch.batch = self.transcription_handler._handle_transcription_batch(self.training_batch.batch)
            self.batch_processor._extract_batch_embeddings_as_dict(i, self.batch_size, self.training_batch.batch)

        BatchConcatenator().concatenate_batches(self.embeddings_path + self.phonetic_model.__class__.__name__ 
        + '_' + args.file_path.split('/')[-1] + '.pkl', self.embeddings_path, self.length // self.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phonetic Embeddings Fast Visualization/Evaluation Script')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for training the embeddings')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--secondary_phonetic_path', type=str, default='', help='Path to the phonetic dataset file in csv (a,b,c columns)')
    parser.add_argument('--embeddings_path', type=str, required=True, help='File to save the extracted embeddings')
    parser.add_argument('--p2v_model', type=str, default='', help='Path to the pretrained phonetic embeddings model')
    parser.add_argument('--load_last_batch', type=str, required=True, help='Load the last batch of embeddings (true/false)')
    parser.add_argument('--phonetic_model', type=str, required=True, help='Phonetic model to use for extracting embeddings')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for extracting embeddings')

    args = parser.parse_args()

    processor = PhoneticEmbeddingsProcessor(args)
    processor.extract_embeddings()
