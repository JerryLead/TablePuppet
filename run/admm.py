from types import SimpleNamespace

import wandb
import argparse
import yaml
from loguru import logger
import sys
import time
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from dataset import MIMIC, Yelp, MovieLens_1M, MIMIC_joint
from client.admm import Worker
from server.admm import Server


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    global args
    start_time = time.time()

    run = wandb.init(config=args, project=args.project)
    config = wandb.config
    tag = (
        "VFL-ADMM"
        if config.simulate_VFL_ADMM
        else f'RFL-ADMM-{"V" if config.client_num == 1 else "H"}'
    )
    run.tags = [tag]
    run.name = f'{tag}-{config.dataset}-{"DP" if config.use_DP else "non-DP"}-{datetime.now().strftime("%m%d%H%M")}'

    wandb.define_metric("epoch")
    wandb.define_metric("exec_t", step_metric="epoch", summary="mean")
    if dataset.task == "classification":
        wandb.define_metric("train/loss", step_metric="epoch", summary="min")
        wandb.define_metric("train/acc", step_metric="epoch", summary="max")
        wandb.define_metric("test/loss", step_metric="epoch", summary="min")
        wandb.define_metric("test/acc", step_metric="epoch", summary="max")
    elif dataset.task == "regression":
        wandb.define_metric("train/rmse", step_metric="epoch", summary="min")
        wandb.define_metric("test/rmse", step_metric="epoch", summary="min")
    if config.use_DP:
        wandb.define_metric("privacy_budget", step_metric="epoch", summary="last")

    logger.info(config)
    server.set_args(args = SimpleNamespace(**dict(config)))
    server.train(config.communication_round)

    end_time = time.time()
    logger.info("[RunTime] Total run time: %.2fs" % (end_time - start_time))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RFL-ADMM")
    parser.add_argument("data_path", type=str, help="Data path")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["MIMIC-III", "Yelp", "MovieLens-1M", "MIMIC-III-joint"],
        help="Name of the dataset",
    )

    parser.add_argument(
        "--simulate_VFL_ADMM",
        type=str2bool,
        default=False,
        help="Whether simulate VFL-ADMM",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["Linear", "MLP"],
        default="Linear",
        help="Model type",
    )

    parser.add_argument(
        "--communication_round", type=int, default=10, help="Communication round"
    )
    parser.add_argument("--rho", type=float, default=1, help="ADMM rho parameter")
    parser.add_argument("--rho_decay", type=float, default=1, help="ADMM rho decay")

    parser.add_argument(
        "--z_update_epoch", type=int, default=100, help="SGD epoch number to update z"
    )
    parser.add_argument(
        "--z_update_lr", type=float, default=1, help="SGD learning rate to update z"
    )

    parser.add_argument(
        "--local_epoch",
        type=int,
        default=10,
        help="SGD epoch number to update local model",
    )
    parser.add_argument(
        "--local_lr",
        type=float,
        default=0.01,
        help="SGD learning rate to update local model",
    )
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=1024,
        help="SGD batch size to update local model",
    )
    parser.add_argument(
        "--local_weight_decay",
        type=float,
        default=1e-5,
        help="SGD weight decay to update local model",
    )

    parser.add_argument(
        "--inside_communication_round",
        type=int,
        default=10,
        help="Communication round inside an organization",
    )
    parser.add_argument(
        "--client_num", type=int, default=1, help="Client num inside an organization"
    )

    parser.add_argument("--use_DP", type=str2bool, default=False, help="Whether use DP")
    parser.add_argument("--DP_delta", type=float, default=1e-5, help="DP delta")
    parser.add_argument(
        "--max_per_sample_clip_norm", default=1, type=float, help="Max norm"
    )
    parser.add_argument(
        "--target_epsilon", default=1, type=float, help="Target Epsilon"
    )

    parser.add_argument(
        "--use_label_DP", type=str2bool, default=False, help="Whether use label DP"
    )
    parser.add_argument(
        "--label_DP_sigma", type=float, default=0.5, help="Label DP noise"
    )

    parser.add_argument("--metric", choices=["AUC", "Accuracy"], default="Accuracy")

    parser.add_argument("--use_GPU", type=str2bool, default=True, help="Wheter use GPU")

    parser.add_argument(
        "--project", default="Table-ADMM", type=str, help="W & B project name"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Whether display debug log output"
    )

    parser.add_argument(
        "--sweep_config",
        type=str,
        default=None,
        help="Path to sweep configuration file",
    )
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep to join")

    args = parser.parse_args()

    logger.configure(
        **{
            "handlers": [
                {
                    "sink": sys.stdout,
                    "format": "{time: YYYY-MM-DD HH:mm:ss} - {message}",
                    "level": "DEBUG" if args.debug else "INFO",
                },
            ],
        }
    )

    logger.info(args)

    if args.dataset == "MIMIC-III":
        dataset = MIMIC()
    elif args.dataset == "Yelp":
        dataset = Yelp()
    elif args.dataset == "MovieLens-1M":
        dataset = MovieLens_1M()
    elif args.dataset == "MIMIC-III-joint":
        dataset = MIMIC_joint()

    table_names = list(dataset.table_name_join_key_mapping.keys())
    workers = [Worker.remote(table_name) for table_name in table_names]

    sweep_id = None
    if args.sweep_config is not None:
        with open(args.sweep_config, "r") as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Loaded sweep configuration at {}".format(args.sweep_config))
            logger.info("Sweep configuration is \n{}".format(sweep_configuration))
        if args.sweep_id is None:
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)
            logger.info("Create new sweep, id is {}".format(sweep_id))
    if args.sweep_id is not None:
        sweep_id = args.sweep_id
        logger.info("Join sweep (id = {})".format(sweep_id))

    server = Server(table_names, workers)
    server.load_data(dataset, args.data_path)
    server.build_mapping()

    if sweep_id is not None:
        wandb.agent(sweep_id, project=args.project, function=main)
    else:
        main()
