from itertools import chain
from src.data import get_corpus
from lm_eval.metrics import perplexity
from torch.utils.data import DataLoader
import torch

import numpy as np
from src.io_wrapper import SegmentRecurrentIOWrapper

from lm_eval.metrics import perplexity
from itertools import chain


def post_process(accum_total_output, task_type):
    for i in range(len(accum_total_output)):
        if task_type == "auto encoding":
            accum_total_output[i] = accum_total_output[i][0]
        elif task_type == "perplexity":
            output = accum_total_output[i]
            output = list(chain.from_iterable(output))
            output = perplexity(output)
            accum_total_output[i] = output
    
    return accum_total_output


@torch.no_grad()
def test_on_task(model, tokenizer, task_type, task_name, num_instance, truncation):

    io_wrapper = SegmentRecurrentIOWrapper(tokenizer, model.chunk_size, truncation)
    task = get_corpus(task_name)
    loader = iter(DataLoader(task, batch_size=1, shuffle=False))

    accum_total_output = []

    for _ in range(num_instance):
        data = next(loader)
        assert "task_type" in data.keys()
        data["task_type"] = task_type

        total_output = []

        for inputs, compute_loss in io_wrapper.wrap(data):
            outputs = model(**inputs)
            if compute_loss is not None:
                result = compute_loss(outputs)
                total_output.append(result)

        accum_total_output.append(total_output)

    accum_total_output = post_process(accum_total_output, task_type)

    result = {
        "task_name": task_name,
        "task_type": task_type,
        "num_instance": num_instance,
        "truncation": truncation,
        "chunk_size": model.chunk_size,
        "avg": float(np.mean(accum_total_output)),
        "max": float(max(accum_total_output)),
        "min": float(min(accum_total_output))
    }

    return result


@torch.no_grad()
def test_on_task_for_rmt(model, tokenizer, task_type, task_name, num_instance, truncation):

    io_wrapper = SegmentRecurrentIOWrapper(tokenizer, model.chunk_size, truncation)
    task = get_corpus(task_name)
    loader = iter(DataLoader(task, batch_size=1, shuffle=False))

    accum_total_output = []

    for _ in range(num_instance):
        data = next(loader)
        assert "task_type" in data.keys()
        data["task_type"] = task_type

        total_output = []
        memory = None

        for inputs, compute_loss in io_wrapper.wrap(data):
            inputs.update({"memory": memory})
            outputs = model(**inputs)
            memory = model.model.update_memory(**inputs)
            if compute_loss is not None:
                result = compute_loss(outputs)
                total_output.append(result)

        accum_total_output.append(total_output)
        model.reset()

    accum_total_output = post_process(accum_total_output, task_type)

    result = {
        "task_name": task_name,
        "task_type": task_type,
        "num_instance": num_instance,
        "truncation": truncation,
        "chunk_size": model.chunk_size,
        "avg": float(np.mean(accum_total_output)),
        "max": float(max(accum_total_output)),
        "min": float(min(accum_total_output))
    }

    return result
