############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from pandas import DataFrame
from seaborn import lineplot
from matplotlib.pyplot import show
############   LOCAL IMPORTS   ###########################
from data_types import DataFrameNames
##########################################################

#TODO: make a line plot where dataset contains both good and bad initialisations and the style is based on the good or bad label (then test for more samples)
class InitialisationVisualiser:
    @staticmethod
    def plot_scores(data_to_plot:DataFrame) -> None:
        lineplot(
            data=data_to_plot,
            x=DataFrameNames.NETWORK_ITERATION,
            y=DataFrameNames.NETWORK_SCORE,
            #style=DataFrameNames.SAMPLE,
            hue=DataFrameNames.NETWORK_NAME,
            #estimator=None
        )
        show()