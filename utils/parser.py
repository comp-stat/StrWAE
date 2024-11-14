import yaml
import argparse

def yaml2list(file_path):
    # Load YAML data
    with open(file_path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

    # Flatten the data into a 1-dimensional list
    flattened_list = [item for sublist in yaml_data.values() for item in sublist]

    return flattened_list

def parse():
    parser = argparse.ArgumentParser(description="WAE Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data-dir", type=str, default="./data", help="path to training image.")
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers.")

    parser.add_argument("--model", type=str, default="StrWAE_GAN",
        help=(
            "choose among the models:"
            + "AAE"
            + "StrWAE_GAN,StrWAE_attr, StrWAE_condgen, StrWAE_fair, HCV"
        )
    )
    parser.add_argument("--dataset", type=str, default="MNIST",
        help="dataset name: MNIST, SVHN, CelebA, EyaleB, VGGFace2, CMNIST"
    )

    """
    learning rate
    """
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="learning rate for encoder-decoder optimizers."
    )
    parser.add_argument("--learning-rate-gan", type=float, default=1e-3,
        help="learning rate for generator (discriminator) optimizers."
    )
    parser.add_argument("--learning-rate-sup", type=float, default=3e-3,
        help="learning rate for classifier optimizers."
    ) # semi-supervised AAE

    parser.add_argument("--optim-beta", type=float, default=0.9,
        help="beta1 for encoder-decoder optimizers."
    )
    parser.add_argument("--optim-gan-beta", type=float, default=0.5,
        help="beta1 for generator (discriminator) optimizers."
    )

    """
    hyperparams: training 
    """
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs for training.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2023, help="random seed for reproducibility.")
    parser.add_argument("--coordinate", type=bool, default=False, help="updates like coordinate update.")

    parser.add_argument("--classifier", type=str, default="./checkpoints/classifiers/mnist_simple_32.pt") # lambda

    """
    hyperparams: model
    """
    parser.add_argument("--conv-layers", type=int, default=3)
    parser.add_argument("--fc-layers", type=int, default=3)
    parser.add_argument("--disc-layers", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--disc-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--linear-bn", type=bool, default=False)

    # network parameters for vggface2 & eyaleb dataset
    parser.add_argument("--enc-block-layers", type=int, default=2)
    parser.add_argument("--dec-block-layers", type=int, default=2)
    parser.add_argument("--enc-scaling-steps", type=int, default=1)
    parser.add_argument("--dec-scaling-steps", type=int, default=1)
    parser.add_argument("--enc-conv-steps", type=int, default=1)
    parser.add_argument("--dec-conv-steps", type=int, default=1)
    parser.add_argument("--enc-skip", type=bool, default=False)
    parser.add_argument("--dec-skip", type=bool, default=False)
    parser.add_argument("--enc-fc-layers", type=int, default=1)
    parser.add_argument("--dec-fc-layers", type=int, default=1)
    
    """
    hyperparams: weight for loss
    """
    parser.add_argument("--lambda-gan", type=float, default=1)
    parser.add_argument("--lambda-disc", type=float, default=1)
    parser.add_argument("--lambda-sup", type=float, default=1)
    parser.add_argument("--lambda-mmd", type=float, default=1)
    parser.add_argument("--lambda-hsic", type=float, default=1)
    parser.add_argument("--lambda-attr", type=float, default=1)
    
    return parser.parse_args()