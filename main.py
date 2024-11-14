import os, importlib
from tqdm import tqdm
from collections import defaultdict
from itertools import cycle
from typing import List, Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance as FID

import wandb
from utils.transform import T
from utils.parser import yaml2list, parse
from utils.dataset import SVHNSearchDataset, VGGFace2_h5, extended_yaleb_pkl, Adult_pkl
from utils.metric import adult_metric

def main(args):
    # start a new wandb run to track this script
    # set the wandb project where this run will be logged
    wandb.init(
        project="strwae",
        group=f"{args.model}_{args.dataset}",
        anonymous="allow"
    )
    wandb.config.update(args)
    wandb.define_metric("gen_fid", summary="min")
    wandb.define_metric("best_epoch", summary="max")
    
    torch.manual_seed(args.random_seed)
    # Default used by PyTorch: highest
    # Faster, but less precise: high
    # Even faster, but also less precise: Medium
    torch.set_float32_matmul_precision("high")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.device("cuda") == device:
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_lower = args.model.lower()
    dataset_lower = args.dataset.lower()
    
    # load dataset
    train_dataset, valid_dataset = load_dataset(dataset_lower, args.data_dir)
    repeat = False # over-sampling labeled data
    sampler_train_labeled = None
    
    if model_lower in ["strwae_semi_cls", "aae"]:
        wandb.define_metric("classification_acc", summary="max")
        wandb.define_metric("conditional_generation_acc", summary="max")

        if dataset_lower in ["mnist"]:
            train_index = yaml2list(f"{args.data_dir}/mnist_indices.yaml")

        elif dataset_lower in ["svhn"]:
            train_index = yaml2list(f"{args.data_dir}/svhn_indices.yaml")
            
        sampler_train_labeled = SubsetRandomSampler(train_index)

        unlabeled_index = list(set(range(len(train_dataset))) - set(train_index))
        sampler_train_unlabeled = SubsetRandomSampler(unlabeled_index)
        
        unlabeled_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler_train_unlabeled,
            drop_last=True
        )
        repeat = True

    labeled_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler_train_labeled,
        shuffle=True if sampler_train_labeled is None else False,
        drop_last=True if dataset_lower != "adult" else False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size if dataset_lower != "vggface2" else args.batch_size // 4,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False if dataset_lower != "vggface2" else True,
        drop_last=True if dataset_lower != "adult" else False,
    )
    
    if dataset_lower != "adult":
        first_batch = next(iter(labeled_train_loader))
        images = first_batch[0][:min(64, args.batch_size)]
        labeled_data = make_grid(images, nrow=8)

        wandb.log({"labeled_images": wandb.Image(labeled_data)})
    
    model, dict_param, dict_lambda = load_model(args)
    model = model.to(device)

    train_modes, valid_modes = load_modes(model_lower, args.coordinate)
    classifier = None

    if model_lower in ["strwae_semi_cls", "aae"]:
        classifier = load_classifier(args.classifier).to(device) # pretrained classifier for cond. gen. acc.
        train_func = train_ssl
        valid_func = valid_ssl
    else:
        train_func = train_usl
        valid_func = valid_usl

    # optimizers
    dict_optim = model.get_optimizers()
    optimizers, schedulers = (
        dict_optim.get("optimizers"),
        dict_optim.get("schedulers")
    )

    z_smpl = torch.randn(32, dict_param.get("latent_dim"))
    z_smpl = z_smpl.to(device)
    first_valid_batch = next(iter(valid_loader))
    x_smpl = first_valid_batch[0].to(device)

    if model_lower in ["strwae_stack_joint"]:
        # y_smpl = the identity of an image (identity-preserving)
        s_smpl = first_valid_batch[2].to(device)
    elif model_lower in ["strwae_stack_cond"]:
        y_smpl = first_valid_batch[1].to(device)
        s_smpl = first_valid_batch[2].to(device)
    else:
        s_smpl = None

    best_genfid, best_acc, best_cond_acc = float("inf"), 0, 0
    best_valid_loss = float("inf")

    # for FID score
    fid = FID(normalize=True).to(device) if dataset_lower != "adult" else None

    # model = torch.compile(model) # pytorch 2.0: speed up

    if model_lower in ["strwae_embedder"]:
        train_early = 100
        valid_early = 100
    elif model_lower in ["strwae_semi_cls", "aae"]:
        train_early = len(unlabeled_train_loader)
        valid_early = len(valid_loader)
    else:
        train_early = len(labeled_train_loader)
        valid_early = len(valid_loader)

    # train & validation
    for epoch in range(1, args.epochs + 1):
        metric_updated = False
        # train
        if repeat:
            tqdm_train_loader = tqdm(
                zip(cycle(labeled_train_loader), unlabeled_train_loader)
            )
        else:
            tqdm_train_loader = tqdm(labeled_train_loader)

        model.train()
        train_loss_epoch = defaultdict(lambda: 0.0)
        train_size = 0
        tqdm_train_loader.set_description(f"Train epoch {epoch}")
        for i, batch in enumerate(tqdm_train_loader):
            if i >= train_early:
                break

            if isinstance(batch[0], list):
                train_size += batch[0][0].size(0)
            else:
                train_size += batch[0].size(0)

            train_loss_iter = train_func(
                model=model,
                batch=batch,
                dict_lambda=dict_lambda,
                dict_loss=train_loss_epoch,
                modes=train_modes,
                optimizers=optimizers,
                device=device,
            )
            tqdm_train_loader.set_postfix(train_loss=train_loss_iter)

        for keys in train_modes:
            for key in keys:
                if schedulers[key] is not None:
                    if type(schedulers[key]).__name__ == "CosineAnnealingWarmUpRestarts":
                        schedulers[key].step(epoch)
                    else:
                        schedulers[key].step()

        wandb.log({"train/" + k: v / train_size for k, v in train_loss_epoch.items()}, step=epoch)
        
        # validation
        model.eval()
        tqdm_valid_loader = tqdm(valid_loader)
        valid_loss_epoch = defaultdict(lambda: 0.0)

        correct = 0
        valid_size = 0
        tqdm_valid_loader.set_description(f"Val epoch {epoch}")
        for i, batch in enumerate(tqdm_valid_loader):
            if i >= valid_early:
                break
                
            valid_size += batch[0].size(0)
            valid_loss_iter, correct_iter = valid_func(
                model=model,
                batch=batch,
                dict_lambda=dict_lambda,
                dict_loss=valid_loss_epoch,
                fid=fid,
                modes=valid_modes,
                device=device
            )
            correct += correct_iter

            tqdm_valid_loader.set_postfix(
                valid_loss=valid_loss_iter,
                acc=100 * correct / valid_size,
            )

        wandb.log({"valid/" + k: v / valid_size for k, v in valid_loss_epoch.items()}, step=epoch)
        valid_acc = 100 * correct / valid_size

        if dataset_lower == "adult": # tabular data
            # Best validation loss
            curr_valid_loss = sum([valid_loss_epoch[key] for key in valid_modes[-1]])
            if curr_valid_loss < best_valid_loss:
                best_valid_loss = curr_valid_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(wandb.run.dir, f"best_loss.pt")
                )
                wandb.run.summary["best_epoch"] = epoch
            
            """
            Compute fairness metric:
                - Get latent vectors z_1
                - Get prediction y_hat or s_hat from z_1
                - Compute a demographic parity (delta_dp)
            """
            train_latent_lst, valid_latent_lst = [], []
            train_y_lst, train_s_lst, valid_y_lst, valid_s_lst = [], [], [], []
            with torch.no_grad():
                for batch in labeled_train_loader:
                    data, label, attr = batch
                    train_y_lst.append(label.squeeze())
                    train_s_lst.append(attr.squeeze())

                    data = data.to(device)
                    label = label.to(device)
                    attr = attr.to(device)
                    ingredient = model.first_operation(
                        x=data, s=attr, y=label
                    ) # z1, z
                    train_latent_lst.append(ingredient[0])
                
                for batch in valid_loader:
                    data, label, attr = batch
                    valid_y_lst.append(label.squeeze())
                    valid_s_lst.append(attr.squeeze())
                    
                    data = data.to(device)
                    label = label.to(device)
                    attr = attr.to(device)
                    ingredient = model.first_operation(
                        x=data, s=attr, y=label
                    ) # z1, z
                    valid_latent_lst.append(ingredient[0])
            
            # Convert to ndarray
            train_z1 = torch.cat(train_latent_lst, dim=0).cpu().numpy()
            valid_z1 = torch.cat(valid_latent_lst, dim=0).cpu().numpy()
            train_y = torch.cat(train_y_lst, dim=0).numpy()
            valid_y = torch.cat(valid_y_lst, dim=0).numpy()
            train_s = torch.cat(train_s_lst, dim=0).numpy()
            valid_s = torch.cat(valid_s_lst, dim=0).numpy()

            fair_metric_dict = adult_metric(train_z1, valid_z1, train_s, valid_s, train_y, valid_y, cls_name="lr")
            for key, value in fair_metric_dict.items():
                wandb.log({f"metric/lr_{key}": value}, step=epoch)
            
            continue # no image metric for the adult dataset
        
        # FID score
        genfid = fid.compute()
        fid.reset()

        if genfid < best_genfid:
            metric_updated = True
            wandb.run.summary["best_gen_fid"] = genfid
            best_genfid = genfid

            # model_script = torch.jit.script(model)
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, f"best_fid_ep{epoch}.pt")
            )

            wandb.log({"metric/fid": genfid}, step=epoch)
            wandb.run.summary["best_epoch"] = epoch
            
        if model_lower in ["strwae_semi_cls", "aae"]:
            # Accuracy of Y
            wandb.log({
                "metric/classification_acc": valid_acc,
            }, step=epoch)

            if valid_acc > best_acc:
                metric_updated = True
                wandb.run.summary["best_acc"] = valid_acc
                best_acc = valid_acc

                torch.save(
                    model.state_dict(),
                    os.path.join(wandb.run.dir, f"best_acc_ep{epoch}.pt")
                )

                # conditional generation acc. by pretrained classifier
                cond_correct = 0
                for batch in tqdm_valid_loader:
                    cond_correct_iter = conditional_accuracy(
                        model, batch, classifier, device
                    )
                    cond_correct += cond_correct_iter
                valid_cond_acc = 100 * cond_correct / valid_size
                print(f"Validation Conditional Accuracy: {valid_cond_acc:.2f}")

                if valid_cond_acc > best_cond_acc:
                    wandb.run.summary["best_cond_acc"] = valid_cond_acc
                    best_cond_acc = valid_cond_acc

                    torch.save(
                        model.state_dict(),
                        os.path.join(wandb.run.dir, f"best_cond_ep{epoch}.pt")
                    )

                wandb.log({
                    "metric/conditional_generation_acc": valid_cond_acc,
                }, step=epoch)

        if metric_updated:
            # reconstruction
            if model_lower in ["strwae_semi_cls", "aae", "strwae_embedder"]:
                recon = model(x_smpl[:32])

            elif model_lower in ["strwae_stack_joint"]:
                recon = model(x_smpl[:32], s_smpl[:32])
                
            x_recon = torch.cat((x_smpl[:32], recon), dim=0).cpu()
            grid = make_grid(x_recon)

            # generation images
            # x -> y_hat; encoding
            # z_smpl, y_hat -> generation; class-preserving generation
            _, model_dependent_value, y_hat = model.encode(x_smpl[:32])
            if model_lower in ["strwae_semi_cls", "aae"]:
                gen_img = model.decode(z_smpl, y_hat)

            elif model_lower in ["strwae_embedder"]:
                # model_dependent_value = s_hat
                gen_img = model.decode(z_smpl, y_hat, model_dependent_value)

            elif model_lower in ["strwae_stack_joint"]:
                gen_img = model.decode(z_smpl, y_hat, s_smpl[:32])

            gen_img = torch.cat((x_smpl[:32], gen_img), dim=0).cpu()
            gen_grid = make_grid(gen_img)

            wandb.log({
                "recon/y_hat": wandb.Image(grid),
                "generation/y_hat": wandb.Image(gen_grid)
            }, step=epoch)
        
    torch.save(model.state_dict(),os.path.join(wandb.run.dir, "last.pt"))
    wandb.finish()


def load_dataset(dataset: str, data_dir: str):
    dataset_lower = dataset.lower()
    transform = T[dataset_lower] if dataset_lower != "adult" else None

    if dataset_lower in ["mnist"]:
        train_dataset = datasets.MNIST(
            data_dir,
            train=True,
            transform=transform,
            download=True
        )
        valid_dataset = datasets.MNIST(
            data_dir,
            train=False,
            transform=transform,
            download=True
        )

    elif dataset_lower in ["svhn"]:
        train_dataset = SVHNSearchDataset(
            data_dir, split="extra", transform=transform, download=True
        )
        valid_dataset = SVHNSearchDataset(
            data_dir, split="test", transform=transform, download=True
        )

    elif dataset_lower in ["adult"]:
        train_dataset = Adult_pkl(data_dir, train=True)
        valid_dataset = Adult_pkl(data_dir, train=False)
    
    elif dataset_lower in ["eyaleb"]:
        train_dataset = extended_yaleb_pkl(data_dir, train=True)
        valid_dataset = extended_yaleb_pkl(data_dir, train=False)
        
    elif dataset_lower in ["vggface2"]:
        train_dataset = VGGFace2_h5(data_dir, train=True, attr=False)
        valid_dataset = VGGFace2_h5(data_dir, train=False, attr=False)
    
    else:
        raise NotImplementedError

    return train_dataset, valid_dataset


def load_model(args: Dict):
    model_lower = args.model.lower()
    dataset_lower = args.dataset.lower()

    if dataset_lower in ["adult"]:
        encoder = getattr(importlib.import_module("networks.base"), "MLPBlock")
        decoder = getattr(importlib.import_module("networks.base"), "MLPBlock")

    else:
        encoder_network = "networks.convnet"
        decoder_network = "networks.convnet"

        if dataset_lower in ["vggface2", "eyaleb"]:
            encoder_network = "networks.convnet2"
            decoder_network = "networks.convnet2"

        encoder = getattr(
            importlib.import_module(encoder_network),
            "Encoder"
        )
        decoder = getattr(
            importlib.import_module(decoder_network),
            "Decoder"
        )
    
    discriminator = getattr(
        importlib.import_module("networks.base"),
        "MLPBlock"
    )
   
    autoencoder = getattr(importlib.import_module("models"), args.model)

    # default: MNIST
    in_channels = 1
    label_dim = 10
    attr_dim = 0
    input_size = 32
    kernel_size = 4

    if dataset_lower in ["svhn"]:
        in_channels = 3
        label_dim = 10
        
    elif dataset_lower in ["adult"]:
        label_dim = 1
        attr_dim = 1
        input_size = 113

    elif dataset_lower in ["eyaleb"]:
        label_dim = 8
        attr_dim = 2
        input_size = 128
        kernel_sizes = [
            [5, 5, 5, 3, 3], # encoder
            [3, 3, 3, 5, 5, 5] # decoder
        ]

    elif dataset_lower in ["vggface2"]:
        in_channels = 3
        label_dim = 64
        attr_dim = 7
        input_size = 128
        kernel_sizes = [
            [5, 5, 5, 5, 5, 3, 3, 3],
            [5, 5, 5, 5, 5, 3, 3, 3]
        ]

    dict_param = {
        "hidden_size": args.hidden_size,
        "latent_dim": args.latent_dim,
        "linear_bn": args.linear_bn,
        "learning_rate": args.learning_rate,
        "label_dim": label_dim,
        "input_size": input_size,
        "encoder": encoder,
        "decoder": decoder,
    }
    
    dict_lambda = dict()

    if model_lower in ["strwae_semi_cls", "aae"]:

        # parameters dictionary
        dict_param.update({
            "in_channels": in_channels,
            "base_channels": args.base_channels,
            "conv_layers": args.conv_layers,
            "fc_layers": args.fc_layers,
            "kernel_size": kernel_size,
            "learning_rate_gan": args.learning_rate_gan,
            "discriminator": discriminator,
            "disc_size": args.disc_size,
            "disc_layers": args.disc_layers,
        })

        # lambda dictionray
        dict_lambda.update({
            "reconstruction": 1.0,
            "supervised": args.lambda_sup,
            "mmd": args.lambda_mmd,
            "generator": args.lambda_gan,
            "discriminator": args.lambda_disc,
            "hsic": args.lambda_hsic
        })

        if model_lower in ["strwae_semi_cls"]:
            dict_param.update({
                "optim_beta": args.optim_beta,
                "optim_gan_beta": args.optim_gan_beta,
            })
            dict_lambda.update({
                "mmd": args.lambda_mmd,
                "hsic": args.lambda_hsic
            })
        else:
            dict_param.update({
                "learning_rate_sup": args.learning_rate_sup,
            })

    elif model_lower in ["strwae_embedder"]:
        dict_param.update({
            "in_channels": in_channels,
            "base_channels": args.base_channels,
            "kernel_sizes": kernel_sizes,
            "block_layers": [args.enc_block_layers, args.dec_block_layers],
            "scaling_steps": [args.enc_scaling_steps, args.dec_scaling_steps],
            "conv_steps": [args.enc_conv_steps, args.dec_conv_steps],
            "skip": [args.enc_skip, args.dec_skip],
            "fc_layers": [args.enc_fc_layers, args.dec_fc_layers],
            "learning_rate_gan": args.learning_rate_gan,
            "discriminator": discriminator,
            "disc_size": args.disc_size,
            "disc_layers": args.disc_layers,
            "attr_dim": attr_dim
        })

        dict_lambda.update({
            "reconstruction": 1.0,
            "hsic": args.lambda_hsic,
            "generator": args.lambda_gan,
            "discriminator": args.lambda_disc
        })

    elif model_lower in ["strwae_stack_joint"]:
        dict_param.update({
            "in_channels": in_channels,
            "base_channels": args.base_channels,
            "kernel_sizes": kernel_sizes,
            "block_layers": [args.enc_block_layers, args.dec_block_layers],
            "scaling_steps": [args.enc_scaling_steps, args.dec_scaling_steps],
            "conv_steps": [args.enc_conv_steps, args.dec_conv_steps],
            "skip": [args.enc_skip, args.dec_skip],
            "fc_layers": [args.enc_fc_layers, args.dec_fc_layers],
            "learning_rate_gan": args.learning_rate_gan,
            "discriminator": discriminator,
            "disc_size": args.disc_size,
            "disc_layers": args.disc_layers,
            "attr_dim": attr_dim
        })

        dict_lambda.update({
            "reconstruction": 1.0,
            "generator": args.lambda_gan,
            "hsic": args.lambda_hsic,
            "hsic_attr": args.lambda_attr,
            "discriminator": args.lambda_disc
        })

    elif model_lower in ["strwae_stack_cond"]:
        dict_param.update({
            "fc_layers": args.fc_layers,
            "learning_rate_gan": args.learning_rate_gan,
            "discriminator": discriminator,
            "disc_size": args.disc_size,
            "disc_layers": args.disc_layers,
        })
        dict_lambda.update({
            "reconstruction": 1.0,
            "generator": args.lambda_gan,
            "hsic": args.lambda_hsic,
            "hsic_attr": args.lambda_attr,
            "discriminator": args.lambda_disc
        })

    dict_param.update({
        "activation": nn.LeakyReLU
    })
    print(dict_param)
    print(dict_lambda)
    return autoencoder(**dict_param), dict_param, dict_lambda


def load_classifier(pretrained_weights: str):
    model = torch.jit.load(pretrained_weights)
    return model


def load_modes(model_name: str, coordinate: bool = False):
    if model_name in ["strwae_semi_cls"]:
        train_modes = (
            ["discriminator"],
            ["reconstruction", "supervised", "mmd", "generator", "hsic"]
        )
        valid_modes = (
            ["discriminator"],
            ["reconstruction", "supervised", "generator", "hsic"]
        )

    elif model_name in ["aae"]:
        train_modes = (
            ["reconstruction"],
            ["generator"],
            ["discriminator"],
            ["supervised"],
            ["mmd"],
            ["hsic"]
        )
        valid_modes = (
            ["reconstruction"],
            ["generator"],
            ["supervised"],
            ["hsic"]
        )

    elif model_name in ["strwae_embedder"]:
        train_modes = (
            ["discriminator"],
            ["reconstruction", "hsic", "generator"],
        )
        valid_modes = (
            ["discriminator"],
            ["reconstruction", "hsic", "generator"],
        )

    elif model_name in ["strwae_stack_joint", "strwae_stack_cond"]:
        train_modes = (
            ["discriminator"],
            ["reconstruction", "generator", "hsic", "hsic_attr"],
        )
        valid_modes = (
            ["discriminator"],
            ["reconstruction", "generator", "hsic", "hsic_attr"],
        )
    
    if coordinate: # Update by each term
        print("coordinate")
        train_modes = [[x] for y in train_modes for x in y]
        valid_modes = [[x] for y in valid_modes for x in y]
    return train_modes, valid_modes


""" Train Function for Semi-supervised Learning, Unsupervised Learning."""

def train_ssl(
    model: nn.Module,
    batch: List[torch.Tensor],
    dict_lambda: Dict,
    dict_loss: Dict,
    modes: List[str],
    optimizers: List[optim.Optimizer] = None,
    device: str = "cuda"):
    
    (labeled_data, labeled_label), (unlabeled_data, _) = batch
    labeled_data = labeled_data.to(device)
    labeled_label = labeled_label.to(device)
    unlabeled_data = unlabeled_data.to(device)

    loss_iter = 0

    # "reconstruction", "penalty", "gan_penalty", "hsic", "discriminator"
    for keys in modes:
        optimizers[keys[0]].zero_grad()
        weighted_loss = 0

        ingredient = model.first_operation(labeled_data, unlabeled_data)
        for key in keys:
            loss = model.get_losses(
                ingredient=ingredient,
                labeled_x=labeled_data,
                labeled_y=labeled_label,
                unlabeled_x=unlabeled_data,
                mode=key,
            )
            weighted_loss += dict_lambda[key] * loss
            dict_loss[key] += loss * labeled_data.size(0)
        weighted_loss.backward()
        loss_iter += weighted_loss.item()
        optimizers[keys[0]].step()

    return loss_iter
    

def valid_ssl(
    model: nn.Module,
    batch: List[torch.Tensor],
    dict_lambda: Dict,
    dict_loss: Dict,
    modes: List[str],
    fid: FID = None,
    device: str = "cuda"):

    correct = 0
    valid_loss_iter = 0

    with torch.no_grad():
        (labeled_data, labeled_label) = batch
        labeled_data = labeled_data.to(device)
        labeled_label = labeled_label.to(device)

        for keys in modes:
            ingredient = model.first_operation(labeled_data)
            weighted_valid_loss = 0

            for key in keys:
                valid_loss, _, y_hat  = model.get_losses(
                    ingredient=ingredient,
                    labeled_x=labeled_data,
                    labeled_y=labeled_label,
                    mode=key,
                    valid=True,
                )
                weighted_valid_loss += dict_lambda[key] * valid_loss
                dict_loss[key] += valid_loss * labeled_data.size(0)
            valid_loss_iter += weighted_valid_loss.item()

        if labeled_data.shape[1] == 1:
            real_fid = torch.cat([labeled_data] * 3, dim=1).to(device)
        else:
            real_fid = labeled_data.to(device)
        fid.update(real_fid, real=True)

        z_prior = torch.randn(
            labeled_data.size(0), model.latent_dim
        ).type_as(labeled_data)
        x_gen = model.decode(z_prior, y_hat)
        
        if x_gen.shape[1] == 1:
            gen_fid = torch.cat([x_gen] * 3, dim=1).to(device)
        else:
            gen_fid = x_gen.to(device)
        fid.update(gen_fid, real=False)

        # classification accuracy
        _, prediction = torch.max(y_hat.data, 1)
        correct += (prediction == labeled_label).sum().item()

    return valid_loss_iter, correct


def conditional_accuracy(
    model: nn.Module,
    batch: List[torch.Tensor],
    classifier: nn.Module = None,
    device: str = "cuda"):

    cond_correct = 0

    with torch.no_grad():
        (labeled_data, labeled_label) = batch
        labeled_data = labeled_data.to(device)
        labeled_label = labeled_label.to(device)

        _, _, y_hat = model.encode(x=labeled_data)

        z_prior = torch.randn(
            labeled_data.size(0), model.latent_dim
        ).type_as(labeled_data)
        x_gen = model.decode(z_prior, y_hat) # class-preserving generation
        
        # conditional generation accuracy (y_hat)
        output = classifier(x_gen).to(device)
        _, cond_prediction = torch.max(output.data, 1)
        cond_correct += (cond_prediction == labeled_label).sum().item()

    return cond_correct


def train_usl(
    model: nn.Module,
    batch: List[torch.Tensor],
    dict_lambda: Dict,
    dict_loss: Dict,
    modes: List[str],
    optimizers: List[optim.Optimizer] = None,
    device: str = "cuda"):
    
    batch = [x.to(device) for x in batch]
    if len(batch) == 2: # vggface2
        data, _ = batch
    else:
        data, label, attr = batch

    loss_iter = 0

    model_lower = type(model).__name__.lower()
    first_op_args = {"x": data}
    if model_lower == "strwae_stack_cond":
        first_op_args.update({"s": attr, "y": label})
    ingredient = model.first_operation(**first_op_args)

    loss_args = {"ingredient": ingredient, "x": data, "mode": None}
    if model_lower == "strwae_stack_joint":
        loss_args.update({"s": attr})
    elif model_lower == "strwae_stack_cond":
        loss_args.update({"s": attr, "y": label})

    for keys in modes:
        optimizers[keys[0]].zero_grad()
        weighted_loss = 0

        for key in keys:
            loss_args.update({"mode": key})
            loss = model.get_losses(**loss_args)
            weighted_loss += dict_lambda[key] * loss
            dict_loss[key] += loss * data.size(0)
        
        weighted_loss.backward()
        loss_iter += weighted_loss.item()
        optimizers[keys[0]].step()

    return loss_iter

def valid_usl(
    model: nn.Module,
    batch: List[torch.Tensor],
    dict_lambda: Dict,
    dict_loss: Dict,
    modes: List[str],
    fid: FID = None,
    device: str = "cuda"):
    
    valid_loss_iter = 0

    with torch.no_grad():
        batch = [x.to(device) for x in batch]
        if len(batch) == 2: # vggface2
            data, _ = batch
        else:
            data, label, attr = batch

        model_lower = type(model).__name__.lower()
        first_op_args = {"x": data}
        if model_lower == "strwae_stack_cond":
            first_op_args.update({"s": attr, "y": label})
        ingredient = model.first_operation(**first_op_args)
        
        loss_args = {"ingredient": ingredient, "x": data, "mode": None, "valid": True}
        if model_lower == "strwae_stack_joint":
            loss_args.update({"s": attr})
        elif model_lower == "strwae_stack_cond":
            loss_args.update({"s": attr, "y": label})

        for keys in modes:
            weighted_valid_loss = 0
            
            for key in keys:
                loss_args.update({"mode": key})
                valid_loss, s_hat, y_hat = model.get_losses(**loss_args)
                
                weighted_valid_loss += dict_lambda[key] * valid_loss
                dict_loss[key] += valid_loss * data.size(0)
            valid_loss_iter += weighted_valid_loss.item()

        if len(data.shape) == 4: # for image data (B, C, H, W)
            if data.shape[1] == 1:
                real_fid = torch.cat([data] * 3, dim=1).to(device)
            else:
                real_fid = data.to(device)
            fid.update(real_fid, real=True)

            z_prior = torch.randn(
                data.size(0), model.latent_dim
            ).type_as(data)
            x_gen = model.decode(z_prior, y_hat, s_hat)
            
            if x_gen.shape[1] == 1:
                gen_fid = torch.cat([x_gen] * 3, dim=1).to(device)
            else:
                gen_fid = x_gen.to(device)
            fid.update(gen_fid, real=False)

    return valid_loss_iter, 0


if __name__ == "__main__":
    args = parse()

    main(args)
