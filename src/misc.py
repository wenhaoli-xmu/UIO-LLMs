import math
from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.modifiers import get_modifier
from src.io_wrapper import SegmentRecurrentIOWrapper
from src.data import get_corpus
from src.eval import test_on_task, test_on_task_for_rmt
from functools import partial

from torch.utils.data import DataLoader
import json


class Saver:
    def __init__(self, model, save, **kwargs):
        self.iter = 0
        self.save = save
        self.model = model

    def step(self):
        self.iter += 1
        if self.iter % self.save == 0:
            self.model.save_checkpoint()
            print("ckp saved", flush=True)


class Evaluator:
    def __init__(self, model, tokenizer, eval, tasks, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.eval = eval
        self.tasks = tasks
        self.iter = 0


    def evaluate(self):
        for task in self.tasks:
            result = test_on_task(self.model, self.tokenizer, **task)
            print(json.dumps(result, indent=4))


    def step(self):
        self.iter += 1
        if self.iter % self.eval == 0:
            self.evaluate()


class RMTEvaluator(Evaluator):
    def evaluate(self):
        for task in self.tasks:
            result = test_on_task_for_rmt(self.model, self.tokenizer, **task)
            print(json.dumps(result, indent=4))


class OptimAnalyzer:
    def __init__(
            self, 
            params: List[torch.Tensor], 
            only_stats: bool = True
        ):
        self.grad_norm = []
        self.avg_grad_norm = []
        self.max_grad_norm = []

        self.grad_norm_ratio = []
        self.avg_grad_norm_ratio = []
        self.max_grad_norm_ratio = []

        self.params = params
        self.only_stats = only_stats
    
    @torch.no_grad()
    def step(self):
        grad_norm = []
        grad_norm_ratio = []
        for param in self.params:
            if param.grad is not None:
                grad_norm.append(torch.norm(param.grad.data).item())
            else:
                grad_norm.append(0)
            grad_norm_ratio.append(grad_norm[-1] / (torch.norm(param.data).item() + 1e-6))
        
        avg_grad_norm = sum(grad_norm) / len(grad_norm)
        max_grad_norm = max(grad_norm)
        avg_grad_norm_ratio = sum(grad_norm_ratio) / len(grad_norm_ratio)
        max_grad_norm_ratio = max(grad_norm_ratio)

        if not self.only_stats:
            self.grad_norm.append(grad_norm)
            self.grad_norm_ratio.append(grad_norm_ratio)
        
        self.avg_grad_norm.append(avg_grad_norm)
        self.max_grad_norm.append(max_grad_norm)
        self.avg_grad_norm_ratio.append(avg_grad_norm_ratio)
        self.max_grad_norm_ratio.append(max_grad_norm_ratio)

        print(f"gd_norm_avg: {avg_grad_norm}")
        print(f"gd_norm_max: {max_grad_norm}")
        print(f"gd_norm_ratio_avg: {avg_grad_norm_ratio}")
        print(f"gd_norm_ratio_max: {max_grad_norm_ratio}", flush=True)


def _data_generator(dataset_infos, partitions, train_iters):
    for _ in range(train_iters):
        choice = (torch.multinomial(torch.tensor(partitions), num_samples=1).item()
                if len(partitions) > 1 else 0)
        loader, io_wrapper = dataset_infos[choice]
        yield next(loader), io_wrapper


def adjust_lr(optim, step, total, max_lr, min_lr, restart, warmup, plateau):
    for param_group in optim.param_groups:
        param_group['lr'] = lr_scheduler(step, total, warmup=warmup, plateau=plateau, max_lr=max_lr, min_lr=min_lr, restart=restart)
        print(f"adjust lr: {param_group['lr']}")


def lr_scheduler(epoch, total_epochs, warmup, plateau, max_lr, min_lr, restart=20):
    total_epochs /= restart
    epoch = epoch % total_epochs

    if epoch / total_epochs < warmup:
        partial = epoch / int(total_epochs * warmup)
        return partial * (max_lr - min_lr) + min_lr
    elif epoch / total_epochs < warmup + plateau:
        return max_lr
    else:
        epoch -= int(total_epochs * (warmup + plateau))
        total_epochs -= int(total_epochs * (warmup + plateau))
        cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        lr = (max_lr - min_lr) * cos_decay + min_lr
    return lr


def get_torch_dtype(dtype: str):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'fp32':
        return torch.float32
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype '{dtype}'")
    

def get_env_conf(env_conf: str):
    import json
    with open(env_conf, 'r') as f:
        env_conf = json.load(f)
    return env_conf


def get_model_and_tokenizer(model_name, model_dtype, model_method, model_structure, save_ckp, load_ckp, config, device_map, **kwargs):
    from accelerate import dispatch_model
    token = "<your_huggingface_access_token>"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    student_dtype = get_torch_dtype(model_dtype)
    student = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=student_dtype, token=token)
    _, student_modifier = get_modifier(model_method, model_structure)
    student = student_modifier(
        student,
        save_ckp=save_ckp,
        load_ckp=load_ckp,
        config=config
    )
    student.eval()
    student.model = dispatch_model(student.model, device_map=device_map)

    return tokenizer, student


def get_optimizer_and_lr_adjuster(model, max_lr, train_iters, warmup, weight_decay, beta1, beta2, **kwargs):
    # optim = torch.optim.RMSprop(model.ft_params(), lr=max_lr, weight_decay=weight_decay)
    optim = torch.optim.AdamW(model.ft_params(), lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster


def get_data_generator(model, tokenizer, train_iters, corpus, **kwargs):
    partitions = []
    dataset_infos = []
    for dataset in corpus:
        loader = iter(DataLoader(get_corpus(dataset["name"]), batch_size=1, shuffle=True))
        io_wrapper = SegmentRecurrentIOWrapper(tokenizer, model.chunk_size, dataset["truncation"])
        dataset_infos.append((loader, io_wrapper))
        partitions.append(float(dataset["partition"]))

    assert abs(sum(partitions) - 1) < 1e-3

    return _data_generator(dataset_infos, partitions, train_iters)


def get_data_generator_deepspeed(model, tokenizer, corpus, **kwargs):
    assert len(corpus) == 1
    corpus = corpus[0]
    dataset = get_corpus(corpus["name"])
    dataset.init_wrapper(tokenizer, model.chunk_size, corpus["truncation"])
    return DataLoader(dataset)
