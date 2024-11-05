import os
import re
import pandas as pd
from argparse import Namespace
from llm import LLaMA3, gpt_easy
from eval import eval_sts, eval_nli
import argparse
import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--dataset_dir", type=str, default="./empathic-stories-main/data/")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct"
)
parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
parser.add_argument(
    "--prompt_type",
    type=str,
    default="optimized",
    choices=["optimized", "optimized_brief", "original", "original_brief"],
)
parser.add_argument(
    "--story_type", type=str, default="summary", choices=["summary", "full"]
)


args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

default_system_prompt = (
    "You are a helpful assistant, can conduct great emotional analysis"
)


empathic_similarity_prompt = """Rate the extent to which you agree with the statement "the narrators of the two stories would empathize with each other." We define empathy as feeling, understanding, and relating to what another person is experiencing. Note that it is possible to have empathy even without sharing the exact same experience or circumstance. Importantly, for two stories to be empathetically similar, both narrators should be able to empathize with each other (if narrator A’s story was shared in response to narrator B’s story, narrator B would empathize with narrator A and vice versa). Give your answer on a scale from 1-4 (1-not at all, 2-not so much, 3-very much, 4-extremely)

Narrative A: {story_a}

Narrative B: {story_b}

Rate in a scale of 1-4.
Answer: 
""".strip()

brief_empathic_similarity_prompt = """Rate how similar two narratives below are, in a scale from 1-4 (1-not at all, 2-not so much, 3-very much, 4-extremely)

Narrative A: {story_a}

Narrative B: {story_b}

Rate in a scale of 1-4.
Answer: """.strip()

optimized_similarity_system_prompt = """Rate the extent to which you agree with the statement "the narrators of the two stories would empathize with each other." We define empathy as feeling, understanding, and relating to what another person is experiencing. Note that it is possible to have empathy even without sharing the exact same experience or circumstance. Importantly, for two stories to be empathetically similar, both narrators should be able to empathize with each other (if narrator A’s story was shared in response to narrator B’s story, narrator B would empathize with narrator A and vice versa). Give your answer on a scale from 1-4 (1-not at all, 2-not so much, 3-very much, 4-extremely), with 0.5 increments in each level between 1-4 are allowed. Please only return the score without any explanation.""".strip()
brief_optimized_similarity_system_prompt = """Rate how similar two narratives below are, in a scale from 1-4 (1-not at all, 2-not so much, 3-very much, 4-extremely), with 0.5 increments in each level between 1-4 are allowed. Please only return the score without any explanation.""".strip()

optimized_similarity_user_prompt = """
### Narrative A:
{story_a}

### Narrative B:
{story_b}

### Similarity Score:
"""

prompt_settings = {
    "optimized": (
        optimized_similarity_system_prompt,
        optimized_similarity_user_prompt,
        ["1", "1.5", "2", "2.5", "3", "3.5", "4"],
        lambda x: float(x),
        lambda x: x > 2.5,
    ),
    "optimized_brief": (
        brief_optimized_similarity_system_prompt,
        optimized_similarity_user_prompt,
        ["1", "1.5", "2", "2.5", "3", "3.5", "4"],
        lambda x: float(x),
        lambda x: x > 2.5,
    ),
    "original": (
        default_system_prompt,
        empathic_similarity_prompt,
        ["1", "2", "3", "4"],
        lambda x: int(x),
        lambda x: x > 2,
    ),
    "original_brief": (
        default_system_prompt,
        brief_empathic_similarity_prompt,
        ["1", "2", "3", "4"],
        lambda x: int(x),
        lambda x: x > 2,
    ),
}

dataset_name = {
    "pair-train": "PAIRS (train).csv",
    "pair-dev": "PAIRS (dev).csv",
    "pair-test": "PAIRS (test).csv",
    "story-train": "STORIES (train).csv",
    "story-dev": "STORIES (dev).csv",
    "story-test": "STORIES (test).csv",
}.get("pair-" + args.split)

story_column_name = {
    "summary": "story_{}_summary",
    "full": "story_{}",
}.get(args.story_type)


normalize = lambda x: x.replace("\n", " ").strip()

df = pd.read_csv(os.path.join(args.dataset_dir, dataset_name))
user_inputs = []
for c in ["A", "B"]:
    df[story_column_name.format(c)] = df[story_column_name.format(c)].apply(normalize)

system_prompt, user_prompt, guided_output, numericalize_func, bin_func = (
    prompt_settings.get(args.prompt_type)
)


user_inputs = df.apply(
    lambda x: user_prompt.format(
        story_a=x[story_column_name.format("A")],
        story_b=x[story_column_name.format("B")],
    ),
    axis=1,
).tolist()

user_inputs_reverse = df.apply(
    lambda x: user_prompt.format(
        story_a=x[story_column_name.format("B")],
        story_b=x[story_column_name.format("A")],
    ),
    axis=1,
).tolist()

scores = []
for user_input, user_input_reverse in tqdm.tqdm(
    zip(user_inputs, user_inputs_reverse), total=len(user_inputs)
):
    response = gpt_easy(
        user_input,
        system_role=system_prompt,
        model=args.model_name,
        temperature=0.0,
        extra_body={"guided_choice": guided_output},
    )
    score = numericalize_func(eval(response))
    response = gpt_easy(
        user_input_reverse,
        system_role=system_prompt,
        model=args.model_name,
        temperature=0.0,
        extra_body={"guided_choice": guided_output},
    )
    score_reverse = numericalize_func(eval(response))
    scores.append((score, score_reverse))

normalize_name = lambda x: x.replace("/", "_").replace("-", "_")

output_name_file_name = "-".join(
    [
        normalize_name(x)
        for x in [args.model_name, args.prompt_type, args.split, args.story_type]
    ]
)

json.dump(scores, open(f"{args.output_dir}/{output_name_file_name}.record.json", "w"))


perf_reports = {}
for h in [
    "similarity_empathy_human_AGG",
    "similarity_event_human_AGG",
    "similarity_emotion_human_AGG",
    "similarity_moral_human_AGG",
]:
    float_gold = df[h].apply(lambda x: float(numericalize_func(x))).tolist()
    bin_gold = df[h].apply(lambda x: int(bin_func(numericalize_func(x)))).tolist()
    score_type_perf = {}
    for score_type, func in zip(
        ["mean", "standard", "reverse"],
        [lambda x: sum(x) / 2, lambda x: x[0], lambda x: x[1]],
    ):
        predicted_scores = [func(x) for x in scores]
        predicted_classes = [int(bin_func(x)) for x in predicted_scores]

        regression_perf = eval_sts(float_gold, predicted_scores)
        classification_perf = eval_nli(bin_gold, predicted_classes)
        score_type_perf[score_type] = {
            "regression": regression_perf,
            "classification": classification_perf,
        }
    perf_reports[h] = score_type_perf

json.dump(
    perf_reports, open(f"{args.output_dir}/{output_name_file_name}.report.json", "w")
)
