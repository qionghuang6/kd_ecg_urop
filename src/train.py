# Adapted from code originally written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# Gradual warmup scheduler (https://github.com/ildoonet/pytorch-gradual-warmup-lr)

import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.data import collate_into_block
from src.utils import Print
import src.evaluate as evaluate
import src.detection_analysis_3d.calc_patient_roc as calc_patient_roc


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-((x - 15) / 3)))


def distillation_loss(outputs, teacher_outputs):
    # apply softmax to the teacher and student's outputs, and divide by the temperature
    teacher_probs = F.softmax(teacher_outputs, dim=1)
    soft_student_probs = F.log_softmax(outputs, dim=1)

    # calculate the KL divergence loss between the teacher and student's soft predictions
    kl_loss = F.kl_div(soft_student_probs, teacher_probs, reduction="batchmean")

    # return the combined distillation loss
    return kl_loss


def remove_last_layers(model, num_layers=5):
    new_model = torch.nn.Sequential()
    for module in list(model.children())[:-num_layers]:
        new_model.append(module)
    return new_model


LAYER_KL_WEIGHTS = [1.0 / 64, 1.0 / 16, 1.0 / 4, 1.0]


class Trainer:
    """train / eval helper class"""

    def __init__(
        self,
        model,
        data_cfg,
        loss_weights,
        teacher=None,
        alpha=0.0,
        use_sigmoid=False,
        kl_layers=1,
    ):
        self.model = model
        self.chunk_length = data_cfg.chunk_length
        self.chunk_stride = data_cfg.chunk_stride
        self.scored_classes = data_cfg.scored_classes
        self.normal_class = data_cfg.normal_class
        self.class_weight, self.confusion_weight = loss_weights
        self.optim = None
        self.scheduler = None
        self.mixup_flag = None
        self.teacher = teacher
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.kl_layers = kl_layers

        # initialize logging parameters
        self.train_flag = False
        self.test_flag = data_cfg.fold is not None
        self.epoch = 0.0
        self.logger_train = Logger()
        self.logger_val = Logger()
        self.logger_test = Logger()
        self.logger_teacher = Logger()

        if self.teacher is not None:
            self.teacher.eval()

    def train(self, batch, device, r1=False, lead=None):
        # training of the model
        if self.mixup_flag:
            batch = self.mixup(batch)
        batch = set_device(batch, device)

        self.model.train()
        inputs, features, labels = batch
        self.optim.zero_grad()

        r_inputs = inputs
        if r1:
            if lead is not None:
                r_inputs = inputs[:, lead : lead + 1, :]
            else:
                raise ValueError("Lead must be provided for 1-lead training")

        outputs, student_logits = self.model(r_inputs, features)
        loss = -torch.mean(
            labels * F.logsigmoid(outputs) + (1 - labels) * F.logsigmoid(-outputs) * 0.1
        )

        kd_l = 0
        kl_div_losses = []
        bce = float(loss)
        multiplier = 1.0
        if self.teacher is not None:
            with torch.no_grad():
                _outputs_teacher, teacher_logits = self.teacher(inputs, features)
            kd_l = 0
            assert len(student_logits) == 4 and len(teacher_logits) == 4
            for i in range(4):
                layer_to_add = 3 - i
                # print(student_logits[layer_to_add].shape, teacher_logits[layer_to_add].shape)
                kl_div_loss = distillation_loss(
                    student_logits[layer_to_add], teacher_logits[layer_to_add]
                )
                kl_div_losses.append(float(kl_div_loss))
                if i < self.kl_layers:
                    kd_l += self.alpha * kl_div_loss * LAYER_KL_WEIGHTS[layer_to_add]
            # print('kl:', float(kd_l), 'bce:', float(bce))
            multiplier = sigmoid(self.epoch) if self.use_sigmoid else 1.0
            kd_l = kd_l * multiplier
            loss += kd_l
            # print('kl:', float(0), 'bce:', float(loss))

        loss.backward()
        self.optim.step()
        self.scheduler.step()

        # logging
        self.logger_train.update(len(labels), loss.item())

        return (
            kl_div_losses,
            kd_l,
            bce,
            float(loss),
            float(self.scheduler.get_last_lr()[0]),
            float(multiplier),
        )

    def evaluate(self, batch, device, split_idx=None, r1=False, lead=None):
        # evaluation of the model
        batch = collate_into_block(batch, self.chunk_length, self.chunk_stride)
        batch = set_device(batch, device)

        self.model.eval()
        inputs, flags, features, labels = batch

        r_inputs = inputs
        if r1:
            if lead is not None:
                r_inputs = inputs[:, lead : lead + 1, :]
            else:
                r_inputs = inputs[:, 0:1, :]

        with torch.no_grad():
            outputs, _ = self.model(r_inputs, features, flags)
            loss = -torch.mean(
                labels * F.logsigmoid(outputs)
                + (1 - labels) * F.logsigmoid(-outputs) * 0.1
            )

        # logging
        scalar_outputs = torch.sigmoid(outputs)
        binary_outputs = scalar_outputs > 0.5

        if split_idx == "val":
            self.logger_val.update(len(labels), loss.item())
            self.logger_val.keep(labels, scalar_outputs, binary_outputs)
        else:
            self.logger_test.update(len(labels), loss.item())
            self.logger_test.keep(labels, scalar_outputs, binary_outputs)

        # if r1 and self.epoch % 10 == 0:
        #     print("evaluate the teacher as a sanity check")
        #     with torch.no_grad():
        #         outputs, _ = self.teacher(inputs, features, flags)
        #         teacher_loss = -torch.mean(
        #             labels * F.logsigmoid(outputs)
        #             + (1 - labels) * F.logsigmoid(-outputs) * 0.1
        #         )
        #     scalar_outputs = torch.sigmoid(outputs)
        #     binary_outputs = scalar_outputs > 0.5
        #     if split_idx == "val":
        #         self.logger_teacher.update(len(labels), teacher_loss.item())
        #         self.logger_teacher.keep(labels, scalar_outputs, binary_outputs)

    def mixup(self, batch):
        # mixup training
        inputs, features, labels = batch

        lengths = np.array([x.shape[1] for x in inputs])
        lam = np.random.beta(0.2, 0.2)
        p = np.random.permutation(len(lengths))
        inputs = lam * inputs + (1 - lam) * inputs[p]
        features = lam * features + (1 - lam) * features[p]
        labels = lam * labels + (1 - lam) * labels[p]

        return inputs, features, labels

    def save_model(self, save_prefix):
        # save state_dicts to checkpoint """
        if save_prefix is None:
            return
        elif not os.path.exists(save_prefix + "/checkpoints/"):
            os.makedirs(save_prefix + "/checkpoints/", exist_ok=True)

        torch.save(
            self.model.state_dict(), save_prefix + "/checkpoints/%d.pt" % self.epoch
        )

    def load_model(self, checkpoint, output):
        # load state_dicts from checkpoint """
        if checkpoint is None:
            return
        Print("loading a model state_dict from the checkpoint", output)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."):
                k = k[7:]
            state_dict[k] = v
        self.model.load_state_dict(state_dict, strict=False)

    def save_outputs(self, idx, save_prefix):
        # save outputs
        if save_prefix is None:
            return
        if not os.path.exists(save_prefix + "/outputs/"):
            os.makedirs(save_prefix + "/outputs/", exist_ok=True)
        self.logger_test.aggregate()
        np.save(save_prefix + "/outputs/%s_labels.npy" % (idx), self.logger_test.labels)
        np.save(
            save_prefix + "/outputs/%s_scalar_outputs.npy" % (idx),
            self.logger_test.scalar_outputs,
        )
        np.save(
            save_prefix + "/outputs/%s_binary_outputs.npy" % (idx),
            self.logger_test.binary_outputs,
        )

    def set_device(self, device):
        # set gpu configurations
        self.model = self.model.to(device)
        if self.class_weight is not None:
            self.class_weight = torch.from_numpy(self.class_weight).to(device)
        if self.confusion_weight is not None:
            self.confusion_weight = torch.from_numpy(self.confusion_weight).to(device)
        if self.teacher is not None:
            self.teacher.to(device)

    def set_optim_scheduler(self, run_cfg, params, steps_per_epoch):
        # set optim and scheduler for training
        optim, scheduler = get_optim_scheduler(run_cfg, params, steps_per_epoch)
        self.train_flag = True
        self.optim = optim
        self.scheduler = scheduler
        self.mixup_flag = run_cfg.mixup

    def get_headline(self):
        # get a headline for logging
        headline = []
        if self.train_flag:
            headline += ["ep", "split"]
            headline += self.logger_train.get_headline(loss_only=True)
            headline += ["|"]

            headline += ["split"]
            headline += self.logger_val.get_headline(loss_only=False)

        if not self.train_flag or self.test_flag:
            headline += ["|", "split"]
            headline += self.logger_test.get_headline(loss_only=False)

        return "\t".join(headline)

    def log(self, output, writer, idx="test"):
        # logging
        log, log_dict = [], {}

        if self.train_flag:
            self.logger_train.evaluate()
            log += ["%03d" % self.epoch, "train"]
            log += self.logger_train.log
            if writer is not None:
                for k, v in self.logger_train.log_dict.items():
                    if k not in log_dict:
                        log_dict[k] = {}
                    log_dict[k]["train"] = v
            log += ["|"]

            self.logger_val.evaluate(
                self.scored_classes, self.normal_class, self.confusion_weight
            )
            log += ["val"]
            log += self.logger_val.log
            if writer is not None:
                for k, v in self.logger_val.log_dict.items():
                    if k not in log_dict:
                        log_dict[k] = {}
                    log_dict[k]["val"] = v

        if not self.train_flag or self.test_flag:
            self.logger_test.evaluate(
                self.scored_classes, self.normal_class, self.confusion_weight
            )
            log += ["|", idx]
            log += self.logger_test.log
            if writer is not None:
                for k, v in self.logger_test.log_dict.items():
                    if k not in log_dict:
                        log_dict[k] = {}
                    log_dict[k][idx] = v

        Print("\t".join(log), output)
        if writer is not None:
            for k, v in log_dict.items():
                writer.add_scalars(k, v, self.epoch)
            writer.flush()

        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_train.reset()
        self.logger_val.reset()
        self.logger_test.reset()
        self.logger_teacher.reset()


class Logger:
    """Logger class"""

    def __init__(self):
        self.total = 0.0
        self.loss = 0.0
        self.labels = []
        self.scalar_outputs = []
        self.binary_outputs = []
        self.log = []
        self.log_dict = {}

    def update(self, total, loss):
        # update logger for current mini-batch
        self.total += total
        self.loss += loss * total

    def keep(self, labels, scalar_outputs, binary_outputs):
        # keep labels and outputs for future computations
        self.labels.append(labels.cpu().detach().numpy())
        self.scalar_outputs.append(scalar_outputs.cpu().detach().numpy())
        self.binary_outputs.append(binary_outputs.cpu().detach().numpy())

    def get_loss(self):
        # get current averaged loss
        loss = self.loss / self.total if self.total > 0 else 0
        return loss

    def get_headline(self, loss_only):
        # get headline
        headline = ["loss"]
        if not loss_only:
            headline += ["acc", "f1", "auroc", "auprc", "cs"]

        return headline

    def evaluate(self, scored_classes=None, normal_class=None, confusion_weight=None):
        # compute evaluation metrics
        self.aggregate()
        metrics = ["loss"]
        evaluations = [self.get_loss()]
        a = None
        # if scored_classes is not None:
        if True:
            # metrics += ["acc", "f1", "auroc", "auprc", "cs"]
            metrics += ["acc", "f1", "auroc", "auprc"]
            evaluations += [
                evaluate.compute_accuracy(self.labels, self.binary_outputs),
                evaluate.compute_f_measure(self.labels, self.binary_outputs)[0],
                *evaluate.compute_auc(self.labels, self.scalar_outputs)[:2],
                # evaluate.compute_challenge_metric(confusion_weight.cpu().numpy(), self.labels, self.binary_outputs,
                #                                   scored_classes, normal_class)
            ]
            a = evaluate.compute_tier_aucs(self.labels, self.scalar_outputs)
            b = dict()
            for i, tier in enumerate(["tier1", "tier2", "tier3"]):
                b[tier] = calc_patient_roc.roc_with_ci(
                    self.scalar_outputs[:, i], self.labels[:, i]
                )
        self.log = ["%.4f" % eval for eval in evaluations]
        self.log_dict = {metric: eval for metric, eval in zip(metrics, evaluations)}
        print(self.log_dict)
        return a, b

    def aggregate(self):
        # aggregate kept labels and outputs
        if isinstance(self.labels, list) and len(self.labels) > 0:
            self.labels = np.concatenate(self.labels, axis=0)
        if isinstance(self.scalar_outputs, list) and len(self.scalar_outputs) > 0:
            self.scalar_outputs = np.concatenate(self.scalar_outputs, axis=0)
        if isinstance(self.binary_outputs, list) and len(self.binary_outputs) > 0:
            self.binary_outputs = np.concatenate(self.binary_outputs, axis=0)

    def reset(self):
        # reset logger
        self.total = 0.0
        self.loss = 0.0
        self.labels = []
        self.scalar_outputs = []
        self.binary_outputs = []
        self.log = []
        self.log_dict = {}


def get_optim_scheduler(cfg, params, steps_per_epoch):
    """configure optim and scheduler"""
    optim = torch.optim.Adam(
        [
            {"params": params[0], "weight_decay": cfg.weight_decay},
            {"params": params[1], "weight_decay": 0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=cfg.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.num_epochs,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=100 * steps_per_epoch, eta_min=cfg.learning_rate/100, last_epoch=-1)

    return optim, scheduler


def set_device(batch, device):
    """recursive function for setting device for batch"""
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
