from abc import ABC, abstractmethod
from typing import Any

from gensim.models import Word2Vec

class PhoneticModelTrainer(ABC):

    @abstractmethod
    def train(train_corpus):
        pass

class Phoneme2VecTrainer(PhoneticModelTrainer):
    def __init__(self, corpus, vector_size=20, window=2, sg=1, workers=4) -> None:
        super().__init__()
        self.corpus = corpus
        self.model = Word2Vec(sentences=self.corpus, vector_size=vector_size, window=window, sg=sg, workers=workers)

    def __repr__(self) -> str:
        return (
            f"Phoneme2Vec[0] -> model.wv[0]: {self.model.wv[0][:5]}, "
            f"\ntrain_time: {self.model.total_train_time}, "
            f"\nvectors: {self.model.wv}, "
            f"\ncorpus_count: {self.model.corpus_count}"
        )
        
    def __getitem__(self, index):
        return self.model.wv[index]

    def train(self, epochs=10, return_model = False):
        self.model.train(self.corpus, total_examples=len(self.corpus), epochs=epochs)
        if return_model:
            return self

