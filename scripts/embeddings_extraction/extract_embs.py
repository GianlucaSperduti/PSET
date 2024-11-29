from SemPhonTest.BatchProcessing import TrainingBatch, BatchEmbsExtractor
from data.data_source import DatasetFactory
from phon_utility.save_and_load import BatchConcatenator
from phon_utility.save_and_load import PickleLoader
from embeddings.embeddings_models import SemanticModelFactory
from phon_utility.phon_utility import TrainingDataBatcher, HighestNumberInFolder
from tqdm import tqdm
import argparse

def string_to_bool(input_str):
    lower_str = input_str.lower()
    if lower_str == "true":
        return True
    elif lower_str == "false":
        return False
    else:
        raise ValueError(f"Invalid input: {input_str}")

class EmbeddingsProcessor:
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
        self.w2v_model = [PickleLoader().load(args.w2v_model) if args.w2v_model != '' else ''][0]
        self.model = SemanticModelFactory.create_model(args.model, self.w2v_model)
        self.load_last_batch_bool = string_to_bool(args.load_last_batch)
        self.last_batch = self._calculate_last_batch() if self.load_last_batch_bool else 0
        self.batch_size = int(args.batch_size)

    def _initialize_components(self):
        self.training_batch = TrainingBatch()
        self.batch_processor = BatchEmbsExtractor(self.model, self.embeddings_path, self.batch_size)
        self.last_batch = self._calculate_last_batch() if self.load_last_batch_bool else 0
        return self.training_batch
    
    def _calculate_last_batch(self):
        batch_calculator = HighestNumberInFolder() 
        last_batch = batch_calculator.operate(self.embeddings_path) 
        if last_batch is None: 
            last_batch = 0 
            return last_batch 
        return last_batch - 1 
 
    # Getting the right training_set format (it varies depending on the dataset type -list; or dataset dict-)
    def _get_training_set_length(self):
        tdb = TrainingDataBatcher() 
        tdb.format_data(self.dataset) 
        training_set = tdb.training_set 
        length = tdb.length 
        return training_set, length 
    
    @staticmethod
    def _word_processing_for_dict(batch):
        return [(word, word) for word in batch]

    # Responsibility: loading the batches in case of existing batches. Extract embeddings 
    def extract_embeddings(self):

        # Actually extracting embeddings 

        for i in tqdm(range(self.last_batch * self.batch_size, self.length, self.batch_size), desc="Processing batches"):
            self.training_batch._get_batch(self.batch_size, i, self.training_set)
            self.training_batch.batch = self._word_processing_for_dict(self.training_batch.batch)
            self.batch_processor._extract_batch_embeddings_as_dict(i, self.batch_size, self.training_batch.batch)

        BatchConcatenator().concatenate_batches(self.embeddings_path + self.model.__class__.__name__ 
        + '_' + args.file_path.split('/')[-1] + '.pkl', self.embeddings_path, self.length // self.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embeddings Fast Visualization/Evaluation Script')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset configuration')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--secondary_phonetic_path', type=str, default='', help='Path to the phonetic dataset file in csv (a,b,c columns)')
    parser.add_argument('--embeddings_path', type=str, required=True, help='Output file for embeddings')
    parser.add_argument('--w2v_model', type=str, default='', help='Path to pre-trained word2vec model (if any)')
    parser.add_argument('--model', type=str, required=True, help='Semantic model to use')
    parser.add_argument('--load_last_batch', type=str, required=True, help='Load the last batch (true/false)')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for processing')

    args = parser.parse_args()
    processor = EmbeddingsProcessor(args)
    processor.extract_embeddings()
