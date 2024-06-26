# UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs

## Experiment setup
```bash
conda env create -f environment.yaml
conda activate uiollms
```

Our repository requires flash attention and PyTorch dependencies, which are related to the local environment, and need to be installed manually with suitable versions according to your environment.

## Download Dataset

```bash
cd uiollms

wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/activation-beacon.tar.gz?download=true -O ./raw_data/activation-beacon.tar.gz

cd raw_data
tar -xzvf activation-beacon.tar.gz
cp activation-beacon/pretrain/redpajama-sample.json .
cp activation-beacon/finetune/longalpaca.json .
```

## Pretrain

```bash
cd uiollms
python train.py --env_conf 32x.json
```
**Instructions on `32x.json` Config**

* You can modify the `device_map` field in the `32x.json` file to change the GPU used for model loading. By assigning different GPUs to different modules, you can achieve pipeline parallelism.
* You can adjust the `config/32x.json` configuration file to change the parameters of LoRA fine-tuning, such as chunk size and compression ratio, etc.
* You can modify the `tbptt` field to change the number of computation graphs retained in memory during UIO-TBPTT algorithm execution. A larger value will slightly improve convergence, but increase overhead.
* You can modify the `corpus` field to configure your desired dataset. All supported datasets and their specific configuration methods can be found in the `src/data.py` file. Use the `truncation` field to configure the maximum token count for each dataset, and `partition` to set the instance ratio for the dataset.
* You can modify the `save_ckp` field to set the save path for checkpoints.

## Benchmark

### Language Modeling

```bash
cd uiollms
python test.py --env_conf 32x.json
```

As long as a checkpoint is generated during the training process (controlled by the `save_ckp` and `save` fields), you can use the above command to perform evaluation.

The `test.py` script will automatically search for a `test.json` file in the working directory, which is used to configure the dataset to be evaluated. Each instance in the file has the following format:
```json
{
    "task_type": "perplexity",
    "task_name": "pg19.test.128k", 
    "num_instance": 100,
    "truncation": 99382
}
```

## Citation
```
@article{uiollms
    author = {Wenhao Li, Mingbao Lin, Yunshan Zhong, Shuicheng Yan, Rongrong Ji},
    year = {2024},
    title = {UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs}
    journal = {arXiv}
}
```