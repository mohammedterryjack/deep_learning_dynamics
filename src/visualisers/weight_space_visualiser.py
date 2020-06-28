############   NATIVE IMPORTS  ###########################
from typing import List
############ INSTALLED IMPORTS ###########################
from numpy import zeros
from seaborn import heatmap
from matplotlib.pyplot import show
from pandas import read_pickle
############   LOCAL IMPORTS   ###########################
from projection_models.projection_method import ProjectionMethod
##########################################################

class WeightSpaceVisualiser:
    def __init__(self, projector:ProjectionMethod,data_filename:str) -> None:
        self.data = read_pickle(f"../data/weight_space_experiment/{data_filename}.pkl")
        self.trained_projector = projector(
            training_vectors=self.data["weights"], 
            save_model=False
        )

    def visualise_weight_space(self, resolution:int=30) -> None:
        coordinates = self.trained_projector.reduce_dimensions(
            vectors= self.data["weights"]
        )
        x_coordinates,y_coordinates = list(zip(*coordinates))
        lower_bound_float = min(
            min(x_coordinates),
            min(y_coordinates)
        )
        upper_bound_float = max(
            max(x_coordinates),
            max(y_coordinates)
        )
        scale = resolution / upper_bound_float
        float_to_int = lambda number: int((number + abs(lower_bound_float))*scale)
        lower_bound_int = float_to_int(lower_bound_float)
        upper_bound_int = float_to_int(upper_bound_float)
        print(
            f"""
            scale = {scale}
            lower_bound_float = {lower_bound_float}
            upper_bound_float = {upper_bound_float}
            lower_bound_int = {lower_bound_int}
            upper_bound_int = {upper_bound_int}
            """
        )
        self._show_heatmap(
            size = (upper_bound_int - lower_bound_int) + 1,
            x_coordinates_as_ints = map(float_to_int,x_coordinates),
            y_coordinates_as_ints = map(float_to_int,y_coordinates),
            scores = self.data["scores"]
        )
        
    @staticmethod
    def _show_heatmap(
        size:int,
        x_coordinates_as_ints:List[int],
        y_coordinates_as_ints:List[int],
        scores:List[float]
    ) -> None:
        matrix = zeros(shape=(size,size))

        for x,y,score in zip(
            x_coordinates_as_ints,
            y_coordinates_as_ints,
            scores
        ):
            matrix[x][y] = max(matrix[x][y],score)

        heatmap(data=matrix)
        show()