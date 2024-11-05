# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from contextlib import nullcontext
import numpy as np

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
import trl
from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
import datasets
from datasets import load_dataset, Dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer
import pandas as pd

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM,
)
from dataclasses import dataclass, field
from typing import Optional

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )

optimized_similarity_system_prompt = """
We define empathy as feeling, understanding, and relating to what another person is experiencing.
Note that it is possible to have empathy even without sharing the exact same experience or circumstance.
Your task is to measure the empathic similarity of the given two stories.
Importantly, for two stories to be empathetically similar, both narrators should be able to empathize with each other (if narrator A’s story was shared in response to narrator B’s story, narrator B would empathize with narrator A and vice versa).
Give your answer on a scale from 1-4 (1-not at all, 2-not so much, 3-very much, 4-extremely), with 0.5 increments in each level between 1-4 are allowed.
You should first analyze the two stories, and then, return the score in a JSON object.""".strip()

optimized_similarity_user_prompt = """
## Narrative A:
{story_a}

## Narrative B:
{story_b}
"""

optimized_similarity_response_prompt = """
## Analysis
{analysis}

## Similarity Score:
```
{{"score": {score}}}
```
"""


@dataclass
class DatasetArguments:
    datapath: str = field(default=None, metadata={"help": "Path to the dataset"})
    analysis_path: str = field(
        default=None, metadata={"help": "Path to the analysis dataset"}
    )
    label_in_use: str = field(
        default="empathy", metadata={"help": "Label column in use"}
    )
    story_in_use: str = field(
        default="summary", metadata={"help": "Story column in use"}
    )


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig, DatasetArguments))
    args, training_args, model_config, dataset_args = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.added_tokens_encoder[
        "<|reserved_special_token_0|>"
    ]

    ################
    # Dataset
    ################
    data_path = dataset_args.datapath
    train_df = pd.read_csv(f"{data_path}/PAIRS (train).csv")
    dev_df = pd.read_csv(f"{data_path}/PAIRS (dev).csv")
    test_df = pd.read_csv(f"{data_path}/PAIRS (test).csv")

    process_analysis = lambda d: "\n\n".join(
        d.split("\n\n")[
            np.argmax(
                list(map(lambda x: "Thematic Similarities" in x, d.split("\n\n")))
            ) :
        ]
    )
    analysis_path = dataset_args.analysis_path
    train_analysis = pd.read_json(
        f"{analysis_path}/train_explanations_{dataset_args.story_in_use}.json"
    ).rename(columns={0: "analysis"})
    train_analysis["analysis"] = train_analysis.apply(
        lambda x: process_analysis(x["analysis"]), axis=1
    )
    dev_analysis = pd.read_json(
        f"{analysis_path}/dev_explanations_{dataset_args.story_in_use}.json"
    ).rename(columns={0: "analysis"})
    dev_analysis["analysis"] = dev_analysis.apply(
        lambda x: process_analysis(x["analysis"]), axis=1
    )
    test_analysis = pd.read_json(
        f"{analysis_path}/test_explanations_{dataset_args.story_in_use}.json"
    ).rename(columns={0: "analysis"})
    test_analysis["analysis"] = test_analysis.apply(
        lambda x: process_analysis(x["analysis"]), axis=1
    )

    label_columns = "similarity_empathy_human_AGG	similarity_event_human_AGG	similarity_emotion_human_AGG	similarity_moral_human_AGG".split()[
        ::-1
    ]
    text_preprocess = lambda x: x.strip().replace("\n", " ")
    # Choose which two columns are used as text pairs

    text_columns = {
        "summary": ["story_A_summary", "story_B_summary"],
        "full": ["story_A", "story_B"],
    }.get(dataset_args.story_in_use)
    label_column = {
        "empathy": ["similarity_empathy_human_AGG"],
    }.get(dataset_args.label_in_use)

    score_conversion_funcs = {
        "none": lambda x: x,
        "original_paper": lambda x: x / 4,
        "01_continue": lambda x: (x - 1) / 3,
    }
    score_recover_funcs = {
        "none": lambda x: x,
        "original_paper": lambda x: x * 4,
        "01_continue": lambda x: (x * 3) + 1,
    }
    # Choose score conversion method
    score_conversion_in_use = "none"
    score_conversion_func = score_conversion_funcs.get(score_conversion_in_use)
    score_recover_func = score_recover_funcs.get(score_conversion_in_use)
    # text_columns = ['story_A', 'story_B']

    def create_data(df, text_pps, score_conversion_funcs, analysis_df):
        required_columns = text_columns + label_column
        score_names = [f"score_{i}" for i in range(len(label_column))]
        df = df[required_columns].rename(
            columns={
                k: v
                for k, v in zip(
                    required_columns, ["sentence1", "sentence2"] + score_names
                )
            }
        )
        for i in [1, 2]:
            df[f"sentence{i}"] = df[f"sentence{i}"].apply(text_pps)
        for i in range(len(label_column)):
            df[f"score_{i}"] = score_conversion_funcs(df[f"score_{i}"])
        return pd.concat([df, analysis_df], axis=1)[
            ["sentence1", "sentence2", "analysis"] + score_names
        ]

    train_df = create_data(
        train_df, text_preprocess, score_conversion_func, train_analysis
    )
    dev_df = create_data(dev_df, text_preprocess, score_conversion_func, dev_analysis)
    test_df = create_data(
        test_df, text_preprocess, score_conversion_func, test_analysis
    )

    def promptify(x1, x2, analysis, labels):
        user_input = optimized_similarity_user_prompt.format(
            story_a=x1, story_b=x2
        ).strip()
        label = optimized_similarity_response_prompt.format(
            analysis=analysis, score=score_recover_func(labels[0])
        ).strip()
        model_input = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": optimized_similarity_system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": label},
            ],
            tokenize=False,
        )
        return {"text": model_input}

    def create_dataset(df, concat_reverse=False, shuffle=False):
        score_columns = [f"score_{i}" for i in range(len(label_column))]
        dataset = (
            Dataset.from_pandas(df)
            .map(
                lambda x: tokenizer(
                    promptify(
                        x["sentence1"],
                        x["sentence2"],
                        x["analysis"],
                        [x[s] for s in score_columns],
                    )["text"],
                    add_special_tokens=False,
                )
            )
            .remove_columns(["sentence1", "sentence2", "analysis"] + score_columns)
        )
        if concat_reverse:
            _ = (
                Dataset.from_pandas(df)
                .map(
                    lambda x: tokenizer(
                        promptify(
                            x["sentence1"],
                            x["sentence2"],
                            x["analysis"],
                            [x[s] for s in score_columns],
                        )["text"],
                        add_special_tokens=False,
                    )
                )
                .remove_columns(["sentence1", "sentence2", "analysis"] + score_columns)
            )
            dataset = datasets.concatenate_datasets([dataset, _])
        if shuffle:
            dataset = dataset.shuffle()
        return dataset

    train_dataset = create_dataset(train_df, concat_reverse=True, shuffle=True)
    eval_dataset = create_dataset(dev_df, concat_reverse=False, shuffle=False)
    response_template = tokenizer(
        "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
    )["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    ################
    # Optional rich context managers
    ###############
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the SFTTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=collator,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
