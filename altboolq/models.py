import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched

from transformers import AutoModel


def predict_argmax_mean(outputs, idxs, mask):
    # set padding outputs to 0
    outputs = outputs.masked_fill(mask.unsqueeze(-1), 0.0)

    # take the mean of each class across the sequence
    outputs = torch.mean(outputs, dim=-2)

    # take the argmax of mean across all windows
    _, counts = idxs.unique_consecutive(dim=0, return_counts=True)
    outputs = outputs.split(counts.tolist())
    outputs = [torch.argmax(torch.mean(y, dim=0), dim=0).item() for y in outputs]
    return outputs


class FF(nn.Module):
    def __init__(self, dim, ff_dim, output_dim):
        super().__init__()
        self.ff_linear = nn.Linear(dim, ff_dim)
        self.activation = F.gelu
        self.layer_norm = nn.LayerNorm(ff_dim)
        self.output = nn.Linear(ff_dim, output_dim)

    def forward(self, x):
        x = self.ff_linear(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.output(x)
        return x


class SoftmaxHead(nn.Module):
    def __init__(self, dim, ff_dim, output_dim, ignore_idx=-1):
        super().__init__()
        self.ff = FF(dim, ff_dim, output_dim)
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.ignore_idx = ignore_idx

    def forward(self, x, mask):
        x = self.ff(x)
        return x

    def get_loss(self, z, y, x):
        mask = x.attention_mask
        y = y.masked_fill(mask == 0, self.ignore_idx)
        return self.loss(z.transpose(1, -1), y.transpose(1, -1))

    @staticmethod
    def get_pred(z, x):
        return predict_argmax_mean(z, x.overflow_to_sample_mapping, x.attention_mask)


class BoolQModel(nn.Module):
    def __init__(self, path, head, dropout=None):
        super().__init__()
        if dropout is None:
            self.roberta = AutoModel.from_pretrained(path)
        else:
            self.roberta = AutoModel.from_pretrained(
                path,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )

        config = self.roberta.config
        hidden_size = config.hidden_size
        if head == "softmax":
            self.head = SoftmaxHead(hidden_size, hidden_size, output_dim=2)
        else:
            raise RuntimeError("Unknown model head")

    def forward(self, x):
        mask = x.attention_mask
        x = {
            k: v
            for k, v in x.items()
            if k not in {"offset_mapping", "overflow_to_sample_mapping"}
        }
        x = self.roberta(**x)[0]
        x = self.head(x, mask)
        return x

    def get_loss(self, z, y, x):
        return self.head.get_loss(z, y, x)

    def get_pred(self, z, x):
        return self.head.get_pred(z, x)


def get_linear_warmup_power_decay_scheduler(
    optimizer, warmup_steps, max_num_steps, end_multiplier=0.0, power=1
):
    """Uses a power function a * x^power + b, such that it equals 1.0 at start_step=1
    and the end_multiplier at end_step. Afterwards, returns the end_multiplier forever.
    For the first warmup_steps, linearly increase the learning rate until it hits the power
    learning rate.
    """

    # a = end_lr - start_lr / (end_step ** power - start_step ** power)
    start_multiplier = 1.0
    start_step = 1
    scale = (end_multiplier - start_multiplier) / (
        max_num_steps**power - start_step**power
    )
    # b = start_lr - scale * start_step ** power
    constant = start_multiplier - scale * (start_step**power)

    def lr_lambda(step):
        step = start_step + step
        if step < warmup_steps:
            warmup_multiplier = scale * (warmup_steps**power) + constant
            return float(step) / float(max(1, warmup_steps)) * warmup_multiplier
        elif step >= max_num_steps:
            return end_multiplier
        else:
            return scale * (step**power) + constant

    return sched.LambdaLR(optimizer, lr_lambda)
