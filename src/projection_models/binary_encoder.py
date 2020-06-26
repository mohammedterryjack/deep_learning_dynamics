############   NATIVE IMPORTS  ###########################
from typing import List
############ INSTALLED IMPORTS ###########################
############   LOCAL IMPORTS   ###########################
from projection_models.projection_method import ProjectionMethod
from data_types import Vectors
from neural_grid_search import NeuralGridSearch
##########################################################
class BinaryEncoder(ProjectionMethod):
    """dimensionality reduction via encoding vector as a binary """
    def __init__(self, training_vectors:Vectors,save_model:bool) -> None:
        pass        

    def reduce_dimensions(self,vectors:Vectors) -> Vectors:
        half_len = int(len(vectors[0]) // 2)
        return list(
            map(
                lambda vector: [
                    NeuralGridSearch._convert_binary_vector_to_int(
                        binary_vector=BinaryEncoder._convert_vector_to_binary_vector(
                            float_vector=vector[:half_len]
                        )
                    ),
                    NeuralGridSearch._convert_binary_vector_to_int(
                        binary_vector=BinaryEncoder._convert_vector_to_binary_vector(
                            float_vector=vector[half_len:]
                        )
                    ),
                ],
                vectors
            )
        )
    
    @staticmethod
    def _convert_vector_to_binary_vector(float_vector:List[float],threshold:float=.5) -> List[int]:
        return list(
            map(
                lambda weight: int(bool(weight>threshold)),
                float_vector
            )
        )