#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################
import sys
import os
import numpy as np
from collections import OrderedDict
import uuid
import json

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

import src.config as config
from src.data import (
    get_dataset_from_configs,
    collate_into_list,
    get_loss_weights_and_flags,
)
from src.model.model_utils import get_model, get_profile
from src.train import Trainer
from src.utils import set_seeds

import wandb


################################################################################
#
# Training model function
#
################################################################################

lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(
    data_directory,
    model_directory,
    seed=42,
    alpha=0.01,
    kl_layers=2,
    splitfile="split.json",
    lead=None,
    use_sigmoid=True,
):
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    num_leads = 12 if lead is None else 1

    print("Training for %dleads..." % num_leads)
    print("Params:", seed, alpha, use_sigmoid, kl_layers)

    if num_leads == 1 and lead is not None:
        print(f"KD Training for lead {lead} ({lead_names[lead]})")

    data_cfg = config.DataConfig(
        "config/cv-%dleads.json" % num_leads, splitfile=splitfile
    )
    preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
    model_cfg = config.ModelConfig("config/model.json")
    run_cfg = config.RunConfig("config/run.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Device count:", torch.cuda.device_count())
    print("Device name", torch.cuda.get_device_name(0))

    assert torch.cuda.is_available()
    print("Loading data...")
    set_seeds(seed)

    if num_leads == 1:
        # Loads all the leads so that we can choose which lead to use for training later (Can refactor this to make it more efficient :D )
        data_cfg.leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]

    # data_cfg.filenames = get_filenames_from_split(data_cfg, "all", "train", data_directory)
    dataset_train = get_dataset_from_configs(
        data_cfg, preprocess_cfg, data_directory, split_idx="train"
    )
    # data_cfg.filenames = get_filenames_from_split(data_cfg, "all", "val", data_directory)
    dataset_val = get_dataset_from_configs(
        data_cfg, preprocess_cfg, data_directory, split_idx="val"
    )

    dataset_test = get_dataset_from_configs(
        data_cfg, preprocess_cfg, data_directory, split_idx="test"
    )

    iterator_train = torch.utils.data.DataLoader(
        dataset_train,
        run_cfg.batch_size,  # collate_fn=collate_into_list,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    iterator_val = torch.utils.data.DataLoader(
        dataset_val,
        run_cfg.batch_size,
        collate_fn=collate_into_list,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    iterator_test = torch.utils.data.DataLoader(
        dataset_test,
        run_cfg.batch_size,
        collate_fn=collate_into_list,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    print("training samples: %d" % len(dataset_train))
    print("evaluation samples: %d" % len(dataset_val))
    print("test samples: %d" % len(dataset_test))

    ## initialize a model
    model, params = get_model(model_cfg, num_leads, len(data_cfg.scored_classes))
    # model = model.to(device)
    get_profile(model, num_leads, data_cfg.chunk_length)

    teacher = None
    if num_leads == 1:
        print("we try to use a teacher:")
        model_path = f"new_model/{num_leads}leads_seed{seed}_split{splitfile.split('_')[1].replace('.json', '')}_kl0_best.pt"
        teacher, _ = get_model(model_cfg, 12, len(data_cfg.scored_classes))
        teacher.load_state_dict(torch.load(model_path), strict=False)
        teacher.eval()

    ## setup trainer configurations
    loss_weights_and_flags = get_loss_weights_and_flags(data_cfg, dataset_train)
    trainer = Trainer(
        model,
        data_cfg,
        loss_weights_and_flags,
        teacher=teacher,
        alpha=alpha,
        use_sigmoid=use_sigmoid,
        kl_layers=kl_layers,
    )
    trainer.set_device(device)
    trainer.set_optim_scheduler(run_cfg, params, len(iterator_train))

    tier_aucs = {"tier1": [], "tier2": [], "tier3": []}
    test_tier_aucs = {"tier1": [], "tier2": [], "tier3": []}

    is_one_lead = num_leads == 1

    wandb.init(
        project="ecg-urop-v2",
        config={
            "id": str(uuid.uuid4()),
            "1_lead_transfer_learning": is_one_lead,
            "leads_used": None if (lead is None) else lead_names[lead],
            "num_leads": num_leads,
            "datasets": data_cfg.datasets,
            "learning_rate": run_cfg.learning_rate,
            "epochs": run_cfg.num_epochs,
            "alpha": alpha,
            "use_sigmoid": use_sigmoid,
            "name": f"{num_leads}leads_seed{seed}_split{splitfile.split('_')[1]}_kl{kl_layers}_alpha{alpha}",
            "kl_layers": kl_layers,
            "seed": seed,
        },
    )

    wandb_logs = []

    best_val_auc = 0
    best_epoch = 0

    base_filename = f"{num_leads}leads_seed{seed}_split{splitfile.split('_')[1].replace('.json', '')}_kl{kl_layers}"

    print("Training model...")
    PNC_list = []
    for epoch in range(run_cfg.num_epochs):
        epoch_bce, epoch_kl_no_alpha, epoch_kl, epoch_loss, epoch_lr, epoch_mul = (
            0,
            [0, 0, 0, 0],
            0,
            0,
            0,
            0,
        )
        n = 0
        for B, batch in enumerate(iterator_train):
            kl_no_alphas, kl, bce, loss, lr, mul = trainer.train(
                batch, device, r1=is_one_lead, lead=lead
            )
            epoch_bce += bce
            epoch_kl_no_alpha = [x + y for x, y in zip(epoch_kl_no_alpha, kl_no_alphas)]
            epoch_kl += kl
            epoch_loss += loss
            epoch_lr += lr
            epoch_mul += mul
            n += 1
            if B % 5 == 0:
                print(
                    "# epoch [{}/{}] train {:.1%}".format(
                        epoch + 1, run_cfg.num_epochs, B / len(iterator_train)
                    ),
                    end="\r",
                    file=sys.stderr,
                )
        print(" " * 150, end="\r", file=sys.stderr)

        epoch_lr /= n
        epoch_mul /= n

        for B, batch in enumerate(iterator_val):
            trainer.evaluate(batch, device, "val", r1=is_one_lead, lead=lead)
            if B % 5 == 0:
                print(
                    "# epoch [{}/{}] val {:.1%}".format(
                        epoch + 1, run_cfg.num_epochs, B / len(iterator_val)
                    ),
                    end="\r",
                    file=sys.stderr,
                )
        print(" " * 150, end="\r", file=sys.stderr)

        for B, batch in enumerate(iterator_test):
            trainer.evaluate(batch, device, "test", r1=is_one_lead, lead=lead)
            if B % 5 == 0:
                print(
                    "# epoch [{}/{}] test {:.1%}".format(
                        epoch + 1, run_cfg.num_epochs, B / len(iterator_test)
                    ),
                    end="\r",
                    file=sys.stderr,
                )
        print(" " * 150, end="\r", file=sys.stderr)

        trainer.epoch += 1

        tier_auc_epoch, eval_auc_cis = trainer.logger_val.evaluate(
            trainer.scored_classes, trainer.normal_class, trainer.confusion_weight
        )
        tier_auc_epoch_test, test_auc_cis = trainer.logger_test.evaluate(
            trainer.scored_classes, trainer.normal_class, trainer.confusion_weight
        )

        # Save model at every epoch
        epoch_model_path = os.path.join(
            model_directory,
            f"{num_leads}_leads_model_seed{seed}_split{splitfile.split('_')[1].replace('.json', '')}_kl{kl_layers}_epoch{epoch}.pt",
        )
        torch.save(trainer.model.state_dict(), epoch_model_path)

        # Update best model if current epoch has better validation AUC
        current_val_auc = sum(tier_auc_epoch) / len(tier_auc_epoch)
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch

        PNC_list.append(float(trainer.logger_val.log[-1]))
        print("Epoch: %03d   PNC: %s" % (epoch, trainer.logger_val.log[-1]))
        trainer.log_reset()

        for auc, t in zip(tier_auc_epoch, ["tier1", "tier2", "tier3"]):
            tier_aucs[t].append(auc)
        print("VALIDATION\n", tier_aucs)

        for auc, t in zip(tier_auc_epoch_test, ["tier1", "tier2", "tier3"]):
            test_tier_aucs[t].append(auc)
        print("TEST\n", test_tier_aucs)

        wandb_log = {
            "epoch": epoch,
            "loss": epoch_loss,
            "bce": epoch_bce,
            "kl": epoch_kl,
            "lr": epoch_lr,
            "mul": epoch_mul,
        }

        if is_one_lead:
            for i in range(4):
                layer_to_add = 3 - i
                wandb_log[f"kl_{layer_to_add}"] = epoch_kl_no_alpha[i]

        for k, v in tier_aucs.items():
            wandb_log[f"val_auc_{k}"] = v[-1]
            wandb.define_metric(f"val_auc_{k}", summary="max")

        for k, v in test_tier_aucs.items():
            wandb_log[f"test_auc_{k}"] = v[-1]
            wandb.define_metric(f"test_auc_{k}", summary="max")

        wandb_logs.append(wandb_log)
        wandb.log(wandb_log)

        # Save logs
        log_path = os.path.join("logs", f"{base_filename}.json")
        with open(log_path, "w") as f:
            json.dump(wandb_logs, f)

    # Load the best model
    best_model_path = os.path.join(
        model_directory, f"{base_filename}_epoch{best_epoch}.pt"
    )
    trainer.model.load_state_dict(torch.load(best_model_path))

    # Save the best model as the final model
    final_model_path = os.path.join(model_directory, f"{base_filename}_best.pt")
    torch.save(trainer.model.state_dict(), final_model_path)

    # Clean up intermediate epoch models
    for epoch in range(run_cfg.num_epochs):
        if epoch != best_epoch:
            epoch_model_path = os.path.join(
                model_directory, f"{base_filename}_epoch{epoch}.pt"
            )
            if os.path.exists(epoch_model_path):
                os.remove(epoch_model_path)

    print(
        f"Best model saved from epoch {best_epoch} with validation AUC: {best_val_auc:.4f}"
    )


################################################################################
#
# Running trained model function
#
################################################################################


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording):
    data_cfg, preprocess_cfg, run_cfg, models = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    data_cfg.data = recording.astype("float32")
    data_cfg.header = header.split("\n")
    dataset_val = get_dataset_from_configs(data_cfg, preprocess_cfg, "")
    iterator_val = torch.utils.data.DataLoader(
        dataset_val, 1, collate_fn=collate_into_list
    )

    outputs_list = []
    for model_ in models:
        loss_weights = get_loss_weights_and_flags(data_cfg)
        trainer = Trainer(model_, data_cfg, loss_weights)
        trainer.set_device(device)

        for B, batch in enumerate(iterator_val):
            trainer.evaluate(batch, device)

        outputs = trainer.logger_test.scalar_outputs[0][0]
        outputs_list.append(outputs)

    classes = data_cfg.scored_classes
    num_classes = len(classes)
    labels = np.zeros(num_classes, dtype=int)
    probabilities = np.zeros(num_classes)
    for i in range(num_classes):
        for j in range(len(outputs_list)):
            probabilities[i] += outputs_list[j][i]
        probabilities[i] = probabilities[i] / len(outputs_list)
        if probabilities[i] > 0.5:
            labels[i] = 1

    return classes, labels, probabilities


################################################################################
#
# File I/O functions
#
################################################################################


# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    # load the model from disk
    data_cfg = config.DataConfig("config/cv-%dleads.json" % len(leads))
    preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
    model_cfg = config.ModelConfig("config/model.json")
    run_cfg = config.RunConfig("config/run.json")

    models = []
    model, _ = get_model(model_cfg, len(data_cfg.leads), len(data_cfg.scored_classes))
    checkpoint = torch.load(
        os.path.join(model_directory, "%dleads_model.pt" % len(data_cfg.leads)),
        map_location=torch.device("cpu"),
    )
    state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[7:]
        state_dict[k] = v
    model.load_state_dict(state_dict, strict=False)
    models.append(model)

    eval_list = [data_cfg, preprocess_cfg, run_cfg, models]

    return eval_list


# Define the filename(s) for the trained models. This function is not required. You can change or remove it.


def get_filenames_from_split(data_cfg, dataset_idx, split_idx, data_directory):
    """get filenames from config file and split index"""
    filenames_all = []
    path = data_directory
    for d, dataset in enumerate(data_cfg.datasets):
        if dataset_idx not in ["all", dataset]:
            continue

        if split_idx in ["train", "val", "test"]:
            filenames = data_cfg.split[dataset][split_idx]

        filenames_all += [
            path + "/%s/%s" % (dataset, filename) for filename in filenames
        ]
    return filenames_all


################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.
