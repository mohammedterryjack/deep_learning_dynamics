############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from pandas import DataFrame
from matplotlib.pyplot import show, quiver, annotate
############   LOCAL IMPORTS   ###########################
from data_types import DataFrameNames
##########################################################
#TODO: heatmap of single hidden layers weight space. Get accuracy for each coordinate (find corresponding weight for that coordinate and test network)
#TODO: if initialised in good areas of the weight space - and see how this affects training

#TODO: plot networks on separate graphs when they overlap? (or when layers are tracked separately)
#TODO: plot multiple layers of one network separately on a 2d graph and make the time the 3rd axis 
#TODO: plot x_t by x_t-1 dynamics map to see spatio-temporal learning patterns (especially for layers tracked separately)

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
        seen_names = set()
        for name,score,x,y in zip(
            data[DataFrameNames.NETWORK_NAME],
            data[DataFrameNames.NETWORK_SCORE],
            data[DataFrameNames.X_COORDINATE],
            data[DataFrameNames.Y_COORDINATE]
        ):
            if name not in seen_names:
                seen_names.add(name)
                annotate(
                    name,
                    (x,y),
                    textcoords="offset points", 
                    xytext=(0,30), 
                    ha='center'
                )       

            annotate(
                f"{score:.3f}",
                (x,y),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center'
            )