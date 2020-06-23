############   NATIVE IMPORTS  ###########################
from argparse import ArgumentParser
############ INSTALLED IMPORTS ###########################
from pandas import read_pickle
############   LOCAL IMPORTS   ###########################
from data_loader import DataLoader
from neural_network import DeepNeuralNetworkTrainer
from visualiser import Visualiser
from projection_models.self_organising_map import SelfOrganisingMap
from projection_models.principal_component_analysis import PrincipalComponentAnalysis
from projection_models.autoencoder import AutoEncoder
from projection_models.autoencoder_tsne import AE_Tsne
from projection_models.autoencoder_som import AE_SOM
##########################################################
parser = ArgumentParser()
parser.add_argument("--timestamp",type=str, help="load in a previous datafile using its timestamp - e.g. 11h57m01s20June2020")
parser.add_argument("--average_hidden_layers", action='store_true', default=False, help="pool neural network's hidden layers to reduce network parameters to track")
parser.add_argument("--train_projection_model_on_first_only", action='store_true', default=False, help="projection model is trained on learning of first neural network but is still used to project parameters of all neural networks - default: trained on all neural networks")
parser.add_argument("--projection", type=str, choices=("PrincipalComponentAnalysis","SelfOrganisingMap","AutoEncoder","AE_Tsne","AE_SOM"), default="AutoEncoder", help="reduce dimension of network parameters for plotting")
parser.add_argument("--dataset", type=str, choices=("toy","mnist","omniglot"), default="mnist", help="dataset task for training the neural networks")
parser.add_argument("--iterations", type=int, choices=range(1,100), default=10, help="training iterations")
parser.add_argument("--networks", type=int, choices=range(1,10), default=2, help="number of neural networks")
parser.add_argument("--layer_to_track",type=int,default=None,help="only track learning of a single layer. specify the layer to track")
parser.add_argument("--track_first_n_layers_separately",type=int,default=None,help="track the learning of the first N layers in the neural network separately")
parser.add_argument("--note", type=str, default = "...", help="description added to meta data")
parser.add_argument("--width",type=int,default=8,help="number of neurons in neural network hidden layer")
parser.add_argument("--depth",type=int,default=5,help="number of hidden layers in neural network")
parser.add_argument("--activation",type=str, choices=("identity","logistic","tanh","relu"), default="relu",help="activation function. logistic = sigmoid, identity = none")
parser.add_argument("--dont_save", action='store_true', default=False, help="prevent data from being stored to data directory")
args = parser.parse_args()

if not args.timestamp:
    x,y,c = DataLoader.load(args.dataset)

Visualiser.plot_coordinates(
    data_to_plot= read_pickle(
        f"../data/{args.timestamp}.pkl"
    ) if args.timestamp else DeepNeuralNetworkTrainer.learn(
        training_inputs=x,
        training_outputs=y,
        classes = c,
        projection_model= eval(args.projection),
        train_projection_model_on_first_only=args.train_projection_model_on_first_only,
        number_of_networks=args.networks,
        training_iterations=args.iterations, 
        network_parameters=dict(
            hidden_layer_sizes=[
                args.width for _ in range(args.depth)
            ], 
            activation = args.activation, 
            alpha=1e-4,
            solver='sgd', 
            learning_rate_init=.1
        ),
        layer_to_track=args.layer_to_track,
        track_first_n_layers_separately=args.track_first_n_layers_separately,
        average_hidden_layers=args.average_hidden_layers,
        notes=args.note,
        save_experiment = not args.dont_save
    )
)