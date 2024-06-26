from src.misc import get_model_and_tokenizer
from src.misc import get_env_conf
from src.misc import Evaluator, RMTEvaluator
import argparse, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--use_env_conf_tasks", action="store_true", default=False)
    parser.add_argument("--rmt", action="store_true", default=False)
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    test_conf = get_env_conf("test.json")

    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    ckp_file = "ckp/" + args.env_conf.replace(".json", ".pth")
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dosen't exists")

    evaluator_class = RMTEvaluator if args.rmt else Evaluator

    if args.use_env_conf_tasks:
        evaluator = evaluator_class(model, tokenizer, **env_conf["train"])
    else:
        evaluator = evaluator_class(model, tokenizer, eval=None, tasks=test_conf)
    
    evaluator.evaluate()
