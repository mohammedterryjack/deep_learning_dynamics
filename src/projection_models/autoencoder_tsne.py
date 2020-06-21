############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from sklearn.manifold import TSNE
############   LOCAL IMPORTS   ###########################
from projection_models.autoencoder import AutoEncoder
from data_types import Vectors
##########################################################
    
class AE_Tsne(AutoEncoder):
    """ NonLinear Dimensionality reduction via an Autoencoder (AE) and then Tsne"""

    def __init__(self, training_vectors:Vectors) -> None:
        super().__init__(
            training_vectors=training_vectors,
            hidden_layer_sizes_encoder_only=[1024,512,256],
            projection_dimension=128
        )

    def reduce_dimensions(self,vectors:Vectors) -> Vectors:
        return TSNE(n_components=2).fit_transform(
            self.encoder.predict(vectors)
        )