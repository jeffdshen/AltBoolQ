import copy
from pathlib import Path

import torch

from transformers import AutoTokenizer

from .datasets import to_device
from .models import BoolQModel
from .train import get_boolq_dataset, forward_backward, noop_backward

from transformers import AutoTokenizer


def download_models(wandb, project, name, root="./models/", per_page=100, filters=None):
    api = wandb.Api()
    runs = api.runs(project, filters=filters, per_page=per_page)
    models = {}
    root_dir = Path(root)
    root_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        model = {}
        model["run_id"] = run.id
        model["config"] = {k: v for k, v in run.config.items()}
        run_dir = root_dir / run.id
        with run.file(name).download(root=run_dir) as _:
            pass
        model["path"] = str(run_dir / name)
        models[run.id] = model
    return models


def select_model(models, extract, value):
    result = None
    for _, model in models.items():
        config = model["config"]
        state = extract(config)
        if state == value:
            if model is not None:
                return None
            result = model
    return copy.deepcopy(result)


def predict(df, path, config):
    df = df.copy(deep=True)
    df["label"] = 0

    model = BoolQModel(
        config["model_path"],
        config["head"],
        dropout=config["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    _, loader = get_boolq_dataset(df, tokenizer, config, shuffle=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for example in loader:
            example = to_device(example, device)
            _, example = example[0], example[1:]

            _, pred, _ = forward_backward(
                model, example, config["model_batch_size"], device, noop_backward
            )
            preds += pred

    return preds
