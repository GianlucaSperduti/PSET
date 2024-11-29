from abc import ABC, abstractmethod

from embeddings.embeddings_visualization import PhoneticEmbsProjection

class EmbsExtraction(ABC): 
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def extract_embeddings(self, vocabulary):
        pass

class MatrixFromEmbs(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def similarity_extraction(self, embs_vocabulary):
        pass

class EmbsFastAnalyzer(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def analyze_and_visualize(self):
        pass

# This class extracts phonetic embeddings (matrices) using a phonetic model

class PhoneticEmbsExtraction(EmbsExtraction):
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.phoentic_vocabulary = []

    def extract_embeddings(self, vocabulary, phonetic_dict):
        for vocab in vocabulary:
            phon_vocab = self.embeddings_model.embed(vocab, phonetic_dict)
            self.phoentic_vocabulary.append(phon_vocab)
        return self.phoentic_vocabulary

# This class uses the extraced matrices (from the latter) to create a similarity matrix between all the elements.
# Embeddings must reduce with a reduction technique (PCA the only implemented by now)

class PhoneticSimMatrixFromEmbs(MatrixFromEmbs):
    def __init__(self, similarity_metric) -> None:
        super().__init__()
        self.similarity_metric = similarity_metric

    def similarity_extraction(self, vocabulary, embs):
        similarity = self.similarity_metric(embs)
        similarity_dict = {vocabulary[i]: {vocabulary[j]: similarity[i, j] for j in range(len(vocabulary))} for i in range(len(vocabulary))}
        return similarity_dict

class PhoneticEmbsFastAnalyzer(EmbsFastAnalyzer):
    def __init__(self, vocabulary,reduced_embeddings, similarity_metric, path_to_image) -> None:
        self.vocabulary = vocabulary
        self.similarity_metric = similarity_metric
        self.path_to_image = path_to_image
        self.reduced_embeddings = reduced_embeddings
        self.similarity_dict = {}

    def analyze_and_visualize(self):
        # Create an instance of PhoneticSimMatrixFromEmbs and compute similarity
        sim_matrix = PhoneticSimMatrixFromEmbs(self.similarity_metric)
        self.similarity_dict = sim_matrix.similarity_extraction(self.vocabulary, self.reduced_embeddings)

        # Create an instance of PhoneticMapsCreation and visualize the data
        maps_creator = PhoneticEmbsProjection()
        maps_creator.maps_creation(self.vocabulary, self.reduced_embeddings, self.path_to_image)
        
