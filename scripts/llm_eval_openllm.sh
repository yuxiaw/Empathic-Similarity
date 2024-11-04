# output_dir="../results/llm_eval_openllm"
# dataset_dir="../empathic-stories/data/"
# model_name="meta-llama/Meta-Llama-3-70B-Instruct"
# split=("dev" "test")
# prompt_type=("optimized" "optimized_brief" "original" "original_brief")
# story_type=("summary" "full")


# for s in "${split[@]}"; do
#     for p in "${prompt_type[@]}"; do
#         for t in "${story_type[@]}"; do
#             python ../src/llm_eval_openllm.py \
#                 --output_dir $output_dir \
#                 --dataset_dir $dataset_dir \
#                 --model_name $model_name \
#                 --split $s \
#                 --prompt_type $p \
#                 --story_type $t
#         done
#     done
# done


output_dir="../results/llm_eval_openllm_small"
dataset_dir="../empathic-stories/data/"
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
split=("dev" "test")
prompt_type=("optimized" "optimized_brief" "original" "original_brief")
story_type=("summary" "full")


for s in "${split[@]}"; do
    for p in "${prompt_type[@]}"; do
        for t in "${story_type[@]}"; do
            python ../src/llm_eval_openllm.py \
                --output_dir $output_dir \
                --dataset_dir $dataset_dir \
                --model_name $model_name \
                --split $s \
                --prompt_type $p \
                --story_type $t
        done
    done
done