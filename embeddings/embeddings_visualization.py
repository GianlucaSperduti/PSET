from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

class EmbsProjection(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def maps_creation(self, path_to_image):
        pass

        
class PhoneticEmbsProjection(EmbsProjection):
    def __init__(self) -> None:
        super().__init__()

    def maps_creation(self, vocabulary, reduced_embs, path_to_image):
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embs[:, 0], reduced_embs[:, 1])

        # Add labels to individual data points
        for i, label in enumerate(vocabulary):
            plt.text(reduced_embs[i, 0], reduced_embs[i, 1], label)

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA of Data")
        plt.savefig(path_to_image)



        