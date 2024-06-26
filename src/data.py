import json, re
from torch.utils.data import Dataset
from datasets import load_dataset
import datasets
from .util import DatasetProcessFn, add_eos

from transformers import AutoTokenizer

import numpy as np
from glob import glob

import random


def pack_lm(text: str):
    assert isinstance(text, str)
    return {
        "text": text,
        "task_type": "language modeling"
    }


def pack_ae(prompt: str, response: str):
    assert isinstance(prompt, str) and isinstance(response, str)
    return {
        "prompt": prompt,
        "response": response,
        "task_type": "auto encoding"
    }


def pack_qa(prompt: str, answer: str, train_prompt=True):
    assert isinstance(prompt, str) and isinstance(answer, str) and isinstance(train_prompt, bool)
    return {
        "prompt": prompt,
        "response": answer,
        "train_prompt": train_prompt,
        "task_type": "question answering"
    }



class Data:
    @staticmethod
    def get_process_fn(tokenizer, min_length, max_length, seed=42, with_labels=True):
        patterns = [" ", "\n", "\n\n"]
        rng = np.random.default_rng(seed=seed)
        
        @DatasetProcessFn()
        def process_fn(text=None, input=None, output=None, input_ids=None, labels=None, _index=None, **kwds):
            if text is not None:
                # truncate text for faster processing
                text = text[:max_length * 5]
                inputs = tokenizer(text, max_length=max_length, truncation=True)
                if len(inputs["input_ids"]) < min_length:
                    return None
                inputs["labels"] = inputs["input_ids"].copy()

            elif input is not None:
                input = input.strip()
                output = output.strip()
                # for too long inputs, we truncate first to save encoding time
                if len(input) > 5 * max_length:
                    input = input[:(5 * max_length) // 2] + input[-(5 * max_length) // 2:]

                tokenized_input = tokenizer.encode(input)
                tokenized_output = tokenizer.encode(output, add_special_tokens=False)
                output_length = len(tokenized_output)                
                # some outputs are too long, discard them
                if output_length > max_length:
                    return None

                # truncate from middle
                input_max_length = max_length - output_length
                if len(tokenized_input) > input_max_length:
                    half = int(input_max_length / 2)
                    input = tokenizer.decode(tokenized_input[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_input[-half:], skip_special_tokens=True)
                
                if with_labels:
                    pattern = rng.choice(patterns).tolist()
                    inputs = tokenizer(pattern.join([input, output]))
                    if len(inputs["input_ids"]) < min_length:
                        return None
                    labels = inputs["input_ids"].copy()
                    labels[:-output_length] = [-100 for _ in range(len(labels) - output_length)]
                    inputs["labels"] = labels
                    # NOTE: eos is essential for LLM to learn to stop generation
                    inputs = add_eos(inputs, tokenizer.eos_token_id)
                else:
                    inputs = tokenizer(input)

            elif input_ids is not None:
                if len(input_ids) < min_length or len(input_ids) > max_length:
                    return None
                inputs = {
                    "input_ids": input_ids,
                    "labels": labels,
                }

            else:
                raise NotImplementedError(f"The dataset must contain one of the following fields: 'input' & 'output' (for fine-tuning), 'text' (for pre-training), 'input_ids' & 'labels' for passkey_retrieval.")

            # index is required for evaluation, by default we always add it
            inputs["index"] = _index
            # length is required for grouping
            inputs["length"] = len(inputs["input_ids"])
            return inputs

        return process_fn
    
    @staticmethod
    def prepare_eval_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, max_eval_num=None, cache_dir=None, seed=42):
        if data_files is None:
            return None

        process_fn = Data.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, seed=seed, with_labels=False)

        if max_eval_num is not None:
            dataset = datasets.load_dataset('json', data_files=data_files, split=f'train[:{max_eval_num}]', cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset('json', data_files=data_files, split='train', cache_dir=cache_dir)
        dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, with_indices=True)
        return dataset

    @staticmethod
    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, max_train_num_per_data=None, seed=42, cache_dir=None):
        if data_files is None:
            return None

        if isinstance(data_files, str):
            if "*" in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]
        
        process_fn = Data.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, seed=seed)

        train_datasets = []
        for path in data_files:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=32, batch_size=1280, remove_columns=temp_dataset.column_names)
            if max_train_num_per_data is not None and len(temp_dataset) > max_train_num_per_data:
                temp_dataset = temp_dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
            train_datasets.append(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset



class MiniPile(Dataset):
    """
    minipile.train
    minipile.test
    """

    def __init__(self, split: str):
        self.data = load_dataset("JeanKaddour/minipile", split=split)

    def __getitem__(self, idx):
        return pack_lm(self.data[idx]['text'])

    def __len__(self):
        return len(self.data)


class Wikitext103(Dataset):
    """
    wikitext103.train
    wikitext103.test
    """

    def __init__(self, split: str):
        self.data = load_dataset("wikitext", name="wikitext-103-raw-v1", split=split)
        self.data = "".join((x['text'] for x in self.data))

        requests = []
        heading_pattern = '( \n = [^=]*[^=] = \n )'
        data_split = re.split(heading_pattern, self.data)
        articles = [x for x in data_split[2::2]]
        for article in articles:
            requests.append({"text": article})
        self.requests = requests

        filtered = []
        for request in self.requests:
            if len(request['text']) > 100:
                filtered.append(request)
        self.requests = filtered

    def __getitem__(self, index):
        return pack_lm(self.requests[index]['text'])

    def __len__(self):
        return len(self.requests)
    

class PG19(Dataset):
    """
    pg19.train
    pg19.test
    pg19.train.1m
    pg19.test.1m
    pg19.train.256k
    pg19.test.256k
    pg19.train.128k
    pg19.test.128k
    """

    def __init__(self, split, max_length=None):
        access_token = "<your_huggingface_access_token>"
        self.data = load_dataset("pg19", split=split, token=access_token, trust_remote_code=True)
        
        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        elif max_length == '128k':
            self.maximum = 128 * 1024
        else:
            self.maximum = None

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']

        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)
    

class ProofPile(Dataset):
    """
    proof-pile
    proof-pile.1m
    proof-pile.256k
    """

    def __init__(self, max_length=None):
        with open("raw_data/proof-pile.json", "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        else:
            self.maximum = None

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']
        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)


class CodeParrot(Dataset):
    """
    code-parrot
    code-parrot.1m
    code-parrot.256k
    """

    def __init__(self, max_length=None):
        with open("raw_data/codeparrot.json", "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        else:
            self.maximum = None

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']
        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)


class LongAlpaca(Dataset):
    """
    longalpaca.train.true
    longalpaca.train.false
    longalpaca.test.true
    longalpaca.test.false
    longalpaca.train.false.deepspeed
    """

    def __init__(self, split, train_prompt, deepspeed: str = None):
        access_token = "<your_huggingface_access_token>"
        assert train_prompt in ("true", "false")
        self.train_prompt = True if train_prompt == 'true' else False
        self.data = load_dataset("Yukang/LongAlpaca-12k", split=split, token=access_token)
        
        assert deepspeed is None or deepspeed == 'deepspeed'
        if deepspeed == 'deepspeed':
            self.io_wrapper = None

    def init_wrapper(self, tokenizer, chunk_size, truncation):
        from src.io_wrapper import SegmentRecurrentIOWrapper
        self.io_wrapper = SegmentRecurrentIOWrapper(tokenizer, chunk_size, truncation)
        return self.io_wrapper

    def __getitem__(self, index):
        return pack_qa(self.data[index]['instruction'], self.data[index]['output'], self.train_prompt)
    
    def __len__(self):
        return len(self.data)
    

class MiniPileCopy256(Dataset):
    """
    minipile copy 256.train
    minipile copy 256.test
    """

    def __init__(self, split: str):
        with open("raw_data/minipile_copy_256.json", "r") as f:
            self.data = json.load(f)
        assert split in ("train", "test")
        self.data = self.data[split]

    def __getitem__(self, index):
        return pack_ae(self.data[index]["prompt"], self.data[index]["response"])
    
    def __len__(self):
        return len(self.data)
    

class MiniPileCopy2K(Dataset):
    """
    minipile copy 2k.train
    minipile copy 2k.test
    """

    def __init__(self, split: str):
        with open("raw_data/minipile_copy_2k.json", "r") as f:
            self.data = json.load(f)
        assert split in ("train", "test")
        self.data = self.data[split]

    def __getitem__(self, index):
        return pack_ae(self.data[index]["prompt"], self.data[index]["response"])
    
    def __len__(self):
        return len(self.data)
    

class MiniPileCopy4K(Dataset):
    """
    minipile copy 4k.train
    minipile copy 4k.test
    """

    def __init__(self, split: str):
        with open("raw_data/minipile_copy_4k.json", "r") as f:
            self.data = json.load(f)
        assert split in ("train", "test")
        self.data = self.data[split]

    def __getitem__(self, index):
        return pack_ae(self.data[index]["prompt"], self.data[index]["response"])
    
    def __len__(self):
        return len(self.data)
    

class MemoryUtil2K(Dataset):
    """
    memory util 2k.train
    memory util 2k.test
    """

    def __init__(self, split: str):
        with open("raw_data/minipile_copy_2k.json", "r") as f:
            self.data = json.load(f)
        assert split in ("train", "test")
        self.data = self.data[split]

    def __getitem__(self, index):
        return {
            "text": self.data[index]['prompt'],
            "task_type": "memory utilization"
        }
    
    def __len__(self):
        return len(self.data)


class RedPajamaArxivSample(Dataset):
    """
    redpajama arxiv sample.train.8m
    redpajama arxiv sample.train.1m
    redpajama arxiv sample.train.128k
    """

    def __init__(self, split, max_length=None):
        assert split == 'train'
        
        self.data = []
        data = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
        for sample in data['train']:
            if "arxiv_id" in sample['meta']:
                if max_length == '1m':
                    sample["text"] = sample["text"][:1024 * 1024]
                elif max_length == '128k':
                    sample['text'] = sample['text'][:1024 * 128]
                elif max_length == '8m':
                    sample['text'] = sample['text'][:1024 * 1024 * 8]
                else:
                    raise NotImplementedError
                self.data.append(sample)

    def __getitem__(self, index):
        return pack_lm(self.data[index]["text"])
    
    def __len__(self):
        return len(self.data)


class RedPajamaBookSample(Dataset):
    """
    redpajama book sample.train.8m
    redpajama book sample.train.1m
    redpajama book sample.train.128k
    """

    def __init__(self, split, max_length=None):
        assert split == 'train'
        assert max_length is None or max_length in ("8m", "1m", "128k")

        self.data = []
        with open("raw_data/redpajama_book_sample.jsonl", 'r') as f:
            while len(self.data) < 10000:
                line = f.readline()
                data = json.loads(line)

                if max_length == "1m":
                    data["text"] = data["text"][:1024 * 1024]
                elif max_length == '128k':
                    data['text'] = data['text'][:1024 * 128]
                elif max_length == '8m':
                    data['text'] = data['text'][:1024 * 1024 * 8]
                else:
                    raise NotImplementedError

                self.data.append(data)
                if len(self.data) > 10000:
                    break
            
    def __getitem__(self, index):
        return pack_lm(self.data[index]["text"])
    
    def __len__(self):
        return len(self.data)
    

class RedPajamaV2(Dataset):
    """
    redpajama v2.train
    redpajama v2.test
    """

    def __init__(self, split: str):
        assert split == 'train'
        self.data = load_dataset(
            "togethercomputer/RedPajama-Data-V2", 
            name="sample", 
            split='train', 
            languages=['en'], 
            trust_remote_code=True)

    def __getitem__(self, index):
        return pack_lm(self.data[index]['raw_content'])
    
    def __len__(self):
        return len(self.data)


class NewsQASummationV2(Dataset):
    """
    newsqasumv2
    """

    def __init__(self):
        self.data = load_dataset(
            "glnmario/news-qa-summarization",
            split='train',
            trust_remote_code=True
        )
        self.pg19 = PG19(split='train', max_length='128k')

    def __getitem__(self, index):

        rnd_index = random.randint(0, len(self.pg19) - 1)
        disturb = self.pg19[rnd_index]["text"]

        while len(disturb) < 128 * 1024:
            rnd_index = random.randint(0, len(self.pg19) - 1)
            disturb = self.pg19[rnd_index]["text"]

        return {
            "disturb": disturb,
            "context": self.data[index]["story"],
            "questions": self.data[index]['questions'],
            "answers": self.data[index]['answers'],
            "summation": self.data[index]['summary'],
            "task_type": "news qa summation v2" 
        }
    
    def __len__(self):
        return len(self.data)
    

class NewsQASummarization(Dataset):
    """
    newsqasum.train.lm
    newsqasum.train.distill
    """

    def __init__(self, split, method):
        assert split == 'train'
        assert method in ('distill', 'lm')
        self.method = method
        self.data = load_dataset(
            "glnmario/news-qa-summarization",
            split='train',
            trust_remote_code=True
        )
        self.pg19 = PG19(split='train', max_length='128k')

    def __getitem__(self, index):
        disturb = self.pg19[index % len(self.pg19)]["text"]

        return {
            "disturb": disturb,
            "context": self.data[index]["story"],
            "questions": self.data[index]['questions'],
            "answers": self.data[index]['answers'],
            "summation": self.data[index]['summary'],
            "method": self.method,
            "task_type": "news qa summation" 
        }
    
    def __len__(self):
        return len(self.data)


class BeaconsCorpus(Dataset):
    """
    beacons
    """
    def __init__(self):
        self.rp = []
        with open("raw_data/redpajama-sample.json", 'r') as f:
            for line in f.readlines():
                self.rp.append(json.loads(line))

        self.la = []
        with open("raw_data/longalpaca.json", 'r') as f:
            for line in f.readlines():
                self.la.append(json.loads(line))

    def __getitem__(self, index):
        if index < len(self.rp):
            data = self.rp[index]
            return pack_lm(data['text'])
        else:
            index -= len(self.rp)
            data = self.la[index]
            return pack_qa(data['input'], data['output'], False)
        
    def __len__(self):
        return len(self.rp) + len(self.la)
    

class BeaconsSampled(Dataset):
    """
    beacons sampled.<max_length>
    beacons sampled.<max_length>.no_alpaca
    beacons sampled.<max_length>.with_alpaca
    beacons sampled.<max_length>.no_alpaca.distill
    beacons sampled.<max_length>.with_alpaca.distill
    """
    def __init__(self, max_length, no_alpaca=None, distill=None):
        tokenizer =  AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        max_length = int(max_length)
        assert no_alpaca is None or no_alpaca == 'no_alpaca'

        dataset = (
            ["raw_data/redpajama-sample.json", "raw_data/longalpaca.json"]
            if no_alpaca is None 
            else ["raw_data/redpajama-sample.json"])

        self.dataset = Data.prepare_train_data(
            dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            min_length=1200,
            max_train_num_per_data=200000,
            seed=42,
        )

        assert distill in ('distill', None)
        self.distill = True if distill == 'distill' else False

    def __getitem__(self, index):
        sample = self.dataset[index]

        return {
            "input_ids": sample['input_ids'],
            "labels": sample['labels'],
            "attention_mask": sample['attention_mask'],
            "length": sample['length'],
            "distill": self.distill,
            "task_type": "activation beacons"
        }

    def __len__(self):
        return len(self.dataset)
    

class LongDataCorpus(Dataset):
    """
    longdata-corpus
    longdata-corpus.128k
    longdata-corpus.1m
    """
    def __init__(self, max_length=None):
        if max_length == '128k':
            self.truncation = 128 * 1024
        elif max_length == '1m':
            self.truncation = 1024 * 1024
        else:
            self.truncation = None

        self.data = load_dataset("yuyijiong/LongData-Corpus", split='train', token="<your_huggingface_access_token>")

    def __getitem__(self, index):
        text = self.data[index]['text']
        text = re.sub(r'([^a-zA-Z0-9])\1{4,}', r'\1\1\1\1', text)
        text = re.sub(r'([a-zA-Z0-9]{3,}?)\1+', r'\1', text)
        if self.truncation is not None:
            text = text[:self.truncation]
        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)


class LongDataCorpusCopy(Dataset):

    """
    longdata-corpus copy.128k
    longdata-corpus copy
    """
    def __init__(self, max_length=None):
        self.data = load_dataset("yuyijiong/LongData-Corpus", split='train', token="<your_huggingface_access_token>")

        assert max_length in (None, "128k")
        if max_length == '128k':
            self.max_length = 128 * 1024
        else:
            self.max_length = None   

    def __getitem__(self, index):
        text = self.data[index]['text']
        text = re.sub(r'([^a-zA-Z0-9])\1{4,}', r'\1\1\1\1', text)
        text = re.sub(r'([a-zA-Z0-9]{3,}?)\1+', r'\1', text)
        return {
            "task_type": "longdata copy",
            "text": text[:self.max_length] if self.max_length is not None else text,
        }
    
    def __len__(self):
        return len(self.data)


CORPUS_MAPPING = {

    # language modeling
    "minipile": MiniPile,
    "wikitext103": Wikitext103,
    "pg19": PG19,
    "proof-pile": ProofPile,
    "code-parrot": CodeParrot,
    "longdata-corpus": LongDataCorpus,

    # instruction tuning
    "longalpaca": LongAlpaca,
    
    # auto encoding (copy tasks)
    "minipilecopy256": MiniPileCopy256,
    "minipilecopy2k": MiniPileCopy2K,
    "minipilecopy4k": MiniPileCopy4K,    
    "longdata-corpuscopy": LongDataCorpusCopy,
    
    # redpajama language modeling corpus
    "redpajamaarxivsample": RedPajamaArxivSample,
    "redpajamabooksample": RedPajamaBookSample,
    "redpajamav2": RedPajamaV2,

    # other datasets
    "newsqasum": NewsQASummarization,
    "newsqasumv2": NewsQASummationV2,
    "memoryutil2k": MemoryUtil2K,
    "beacons": BeaconsCorpus,
    "beaconssampled": BeaconsSampled
}


def get_corpus(ds):
    ds = ds.lower().replace(" ", "")
    dataset_name, *args = ds.split(".")

    for name, data_class in CORPUS_MAPPING.items():
        if name == dataset_name:
            return data_class(*args)
    
    raise NotImplementedError(dataset_name)
