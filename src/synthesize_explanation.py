import openai
import pandas as pd
import time
import argparse
import os
import json
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, default="test_vmt")
parser.add_argument("--base_url", type=str, default="http://node16:9876/v1")
parser.add_argument("--data_path", type=str, default="../empathic-stories/data/")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
parser.add_argument("--story_in_use", type=str, default="summary")
parser.add_argument("--output_path", type=str, default="../data/")


args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)


client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)


def call_llama3(messages):
    while True:
        try:
            response = client.chat.completions.create(
                model=args.model, messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(60)


data_path = args.data_path
train_df = pd.read_csv(f"{data_path}/PAIRS (train).csv")
dev_df = pd.read_csv(f"{data_path}/PAIRS (dev).csv")
test_df = pd.read_csv(f"{data_path}/PAIRS (test).csv")


label_columns = "similarity_empathy_human_AGG	similarity_event_human_AGG	similarity_emotion_human_AGG	similarity_moral_human_AGG".split()[
    ::-1
]
text_preprocess = lambda x: x.strip().replace("\n", " ")

story_in_use = args.story_in_use
label_in_use = "empathy"


text_columns = {
    "summary": ["story_A_summary", "story_B_summary"],
    "full": ["story_A", "story_B"],
}.get(story_in_use)
label_column = {
    "empathy": ["similarity_empathy_human_AGG"],
    "event": ["similarity_event_human_AGG"],
    "emotion": ["similarity_emotion_human_AGG"],
    "moral": ["similarity_moral_human_AGG"],
    "all": label_columns,
}.get(label_in_use)


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


def create_data(df, text_pps, score_conversion_funcs):
    required_columns = text_columns + label_column
    score_names = [f"score_{i}" for i in range(len(label_column))]
    df = df[required_columns].rename(
        columns={
            k: v
            for k, v in zip(required_columns, ["sentence1", "sentence2"] + score_names)
        }
    )
    for i in [1, 2]:
        df[f"sentence{i}"] = df[f"sentence{i}"].apply(text_pps)
    for i in range(len(label_column)):
        df[f"score_{i}"] = score_conversion_funcs(df[f"score_{i}"])
    return df


train_df = create_data(train_df, text_preprocess, score_conversion_func)
dev_df = create_data(dev_df, text_preprocess, score_conversion_func)
test_df = create_data(test_df, text_preprocess, score_conversion_func)


system_prompt = """
You are given two stories and an empathic similarity score. The score is in a range from 1 to 4, representing low to high empathic similarity.
Your task is to give an explanation on why the score is assigned with that value, why the two stories have high or low empathic similarity.
Here are some recommanded perspectives that can be addressed in your analysis:

### Thematic Similarities:

- Identify Common Themes: Describe any common themes that both stories share. Consider elements such as setting, plot, character motivations, and moral dilemmas.
- Relevance of Themes to Empathy: Explain how these themes might resonate on an empathetic level with readers or characters within the stories.
### Emotional Content:

- Describe Emotional Tone: Analyze the emotional tone of each story. How do the feelings conveyed in each story align?
- Impact of Emotional Similarity on Empathy: Discuss how similar emotional experiences in the stories might contribute to the empathic similarity score.

### Character Analysis:

- Compare Main Characters: Consider the challenges, desires, and emotional journeys of the main characters in each story.
- Role of Character Experiences in Empathy: Reflect on how the characters' experiences might foster empathy between them or with the audience.

### Narrative Structure:

- Examine Storytelling Techniques: Look at how each story is told. Are there similarities in narrative perspective, pacing, or style?
- Influence of Narrative on Empathy: Suggest how these narrative elements could affect the empathic connection between the stories.
### Overall Empathy Evaluation:

- Synthesize Insights: Combine your observations from the above categories to justify the assigned empathic similarity score.
- Score Justification: Provide a detailed explanation of why the given score is appropriate based on your analysis.

By going through this structured analysis, you can provide a comprehensive explanation of why two stories received their specific empathic similarity score. This exercise not only aids in understanding the nuances of empathy in storytelling but also enhances the model's ability to perform nuanced literary analysis.
"""

user_prompt = """
### Narrative A:
{story_1}

### Narrative B:
{story_2}

### Empathic Similarity:
{score}

### Explanation:
"""


promptify = lambda s1, s2, score: [
    {"role": "system", "content": system_prompt.strip() + "\n"},
    {
        "role": "user",
        "content": user_prompt.format(story_1=s1, story_2=s2, score=score).strip(),
    },
]

train_prompts = train_df.apply(
    lambda x: promptify(x["sentence1"], x["sentence2"], x["score_0"]), axis=1
)
dev_prompts = dev_df.apply(
    lambda x: promptify(x["sentence1"], x["sentence2"], x["score_0"]), axis=1
)
test_prompts = test_df.apply(
    lambda x: promptify(x["sentence1"], x["sentence2"], x["score_0"]), axis=1
)


if not os.path.exists(f"{args.output_path}/train_explanations_{story_in_use}.json"):
    train_explanations = [call_llama3(p) for p in tqdm.tqdm(train_prompts)]
    json.dump(
        train_explanations,
        open(f"{args.output_path}/train_explanations_{story_in_use}.json", "w"),
    )
if not os.path.exists(f"{args.output_path}/dev_explanations_{story_in_use}.json"):
    dev_explanations = [call_llama3(p) for p in tqdm.tqdm(dev_prompts)]
    json.dump(
        dev_explanations,
        open(f"{args.output_path}/dev_explanations_{story_in_use}.json", "w"),
    )

if not os.path.exists(f"{args.output_path}/test_explanations_{story_in_use}.json"):
    test_explanations = [call_llama3(p) for p in tqdm.tqdm(test_prompts)]
    json.dump(
        test_explanations,
        open(f"{args.output_path}/test_explanations_{story_in_use}.json", "w"),
    )
