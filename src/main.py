"""
Purpose
- Run config-driven inference for retrieval-augmented generation (RAG) on multi-hop QA datasets.
- Support multiple retrieval and aggregation strategies and record per-sample diagnostics.

Quick start
- Run: python main.py --config_path path/to/config.json
- The config.json controls: dataset, data_path, method, output_dir, fewshot, sample, retriever, and other generation options.

Supported datasets
- strategyqa, 2wikimultihopqa, hotpotqa, iirc

Supported methods (examples)
- non-retrieval, single-retrieval, fix-length-retrieval, fix-sentence-retrieval, random-sentence-retrieval, token, entity, attn_prob, dragin, seq_confidence

Outputs
- output_dir/config.json — saved config used for the run
- output_dir/output.txt — one JSON line per example with qid, prediction, and optional counters

Location
- Place dataset and config files as referenced by config.json. The script writes outputs into the configured output_dir.

License
- Intended for research and evaluation; adapt and attribute as needed.
"""

import os
import os.path as osp
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from runtime_env import configure_vllm_runtime_env
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC

configure_vllm_runtime_env(os.environ)

from generate import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default='config/Qwen2.5-7B-Instruct/HotpotQA/SeqLevelNoReftGeneQuery.json', help='config path')
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False
    if "use_counter" not in args:
        args.use_counter = True
    if "save_trace" not in args:
        args.save_trace = False
    if "disable_retrieval" not in args:
        args.disable_retrieval = False
    if "experiment_condition" not in args:
        args.experiment_condition = None
    return args


def main():
    args = get_args()
    logger.info(f"{args}")

    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    if 'retriever' in args:
        args.output_dir = osp.join(args.output_dir, args.retriever)
    else:
        args.output_dir = osp.join(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")
    trace_file = None
    if args.save_trace:
        trace_file = open(os.path.join(args.output_dir, "trace.jsonl"), "w")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))

    # 根据 method 选择不同的生成策略
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" \
            or args.method == "fix-sentence-retrieval" \
            or args.method == "random-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "token":
        model = TokenRAG(args)
    elif args.method == "entity":
        model = EntityRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)
    elif args.method == "seq_confidence":
        model = SeqConfidenceRAG(args)
    else:
        raise NotImplementedError

    logger.info("start inference")
    for i in tqdm(range(len(data))):
        last_counter = copy(model.counter)
        batch = data[i]
        infer_ret = model.inference(batch["question"], batch["demo"])
        trace = None
        if isinstance(infer_ret, tuple):
            pred, trace = infer_ret
        else:
            pred = infer_ret
        pred = pred.strip()
        ret = {
            "qid": batch["qid"],
            "prediction": pred,
        }
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        output_file.write(json.dumps(ret)+"\n")
        if trace_file is not None and trace is not None:
            trace["qid"] = batch["qid"]
            trace["question"] = batch["question"]
            trace["final_prediction"] = pred
            trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
