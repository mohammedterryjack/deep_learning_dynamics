############   NATIVE IMPORTS  ###########################
from argparse import ArgumentParser
############ INSTALLED IMPORTS ###########################
from numpy import array
from pandas import read_pickle
############   LOCAL IMPORTS   ###########################
from data_loader import DataLoader
from neural_network import DeepNeuralNetworkTrainer
from visualiser import Visualiser
from self_organising_map import SelfOrganisingMap
from principal_component_analysis import PrincipalComponentAnalysis
from autoencoder import AutoEncoder
from autoencoder_tsne import AE_Tsne
##########################################################
parser = ArgumentParser()
parser.add_argument("--timestamp",type=str, help="load in a previous datafile using its timestamp - e.g. 11h57m01s20June2020")
parser.add_argument("--average_hidden_layers", action='store_true', default=False, help="pool neural network's hidden layers to reduce network parameters to track")
parser.add_argument("--train_projection_model_on_first_only", action='store_true', default=False, help="projection model is trained on learning of first neural network but is still used to project parameters of all neural networks - default: trained on all neural networks")
parser.add_argument("--projection", type=str, choices=("pca","som","ae","ae_tsne"), default="ae", help="reduce dimension of network parameters for plotting")
parser.add_argument("--dataset", type=str, choices=("mnist","omniglot"), default="mnist", help="dataset task for training the neural networks")
parser.add_argument("--iterations", type=int, choices=range(1,100), default=10, help="training iterations")
parser.add_argument("--networks", type=int, choices=range(1,10), default=2, help="number of neural networks")
parser.add_argument("--note", type=str, default = "...", help="description added to meta data")
args = parser.parse_args()

if args.dataset == "mnist":
    x, y, c = DataLoader.mnist()
elif args.dataset == "omniglot":
    x,y,c = DataLoader.omniglot()

Visualiser.plot_coordinates(
    data_to_plot= read_pickle(
        f"data/{args.timestamp}.pkl"
    ) if args.timestamp else DeepNeuralNetworkTrainer.learn(
        training_inputs=x,
        training_outputs=y,
        classes = c,
        projection_model= PrincipalComponentAnalysis if args.projection == "pca" else SelfOrganisingMap if args.projection == "som" else AutoEncoder if args.projection == "ae" else AE_Tsne,
        train_projection_model_on_first_only=args.train_projection_model_on_first_only,
        number_of_networks=args.networks,
        training_iterations=args.iterations, 
        network_parameters=dict(
            hidden_layer_sizes=(8,8,8,8,8), 
            alpha=1e-4,
            solver='sgd', 
            learning_rate_init=.1
        ),
        layer_to_track=None,
        average_hidden_layers=args.average_hidden_layers,
        notes=args.note,
    )
)