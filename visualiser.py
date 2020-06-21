############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from pandas import DataFrame
from matplotlib.pyplot import show, quiver, annotate
############   LOCAL IMPORTS   ###########################
from data_types import DataFrameNames
##########################################################

class Visualiser:
    """ plots learning dynamics """
    @staticmethod
    def plot_coordinates(data_to_plot:DataFrame) -> None:
        Visualiser.annotate_direction(data=data_to_plot)
        Visualiser.annotate_scores(data=data_to_plot)
        show()
    
    @staticmethod
    def annotate_direction(data:DataFrame) -> None:
        for label in data[DataFrameNames.NETWORK_NAME]:
            data_for_this_network = data[data[DataFrameNames.NETWORK_NAME]==label]
            x = data_for_this_network[DataFrameNames.X_COORDINATE].to_numpy()
            y = data_for_this_network[DataFrameNames.Y_COORDINATE].to_numpy()
            quiver(
                x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], 
                scale_units='xy', 
                angles='xy', 
                scale=1, 
                color=data_for_this_network[DataFrameNames.COLOUR]
            )

    @staticmethod
    def annotate_scores(data:DataFrame) -> None:
        for label,x,y in zip(
            data[DataFrameNames.NETWORK_SCORE],
            data[DataFrameNames.X_COORDINATE],
            data[DataFrameNames.Y_COORDINATE]
        ):
            annotate(
                f"{label:.3f}",
                (x,y),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center'
            )