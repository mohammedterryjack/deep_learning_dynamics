############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from pandas import DataFrame
from seaborn import lineplot
from matplotlib.pyplot import show
############   LOCAL IMPORTS   ###########################
from data_types import DataFrameNames
##########################################################

class InitialisationVisualiser:
    @staticmethod
    def plot_scores(data_to_plot:DataFrame) -> None:
        lineplot(
            data=data_to_plot,
            x=DataFrameNames.NETWORK_ITERATION,
            y=DataFrameNames.NETWORK_SCORE,
            style=DataFrameNames.NETWORK_NAME
        )
        show()