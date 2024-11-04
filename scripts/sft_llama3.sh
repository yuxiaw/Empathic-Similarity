# torchrun --nproc-per-node=2 ../src/sft_llama3.py \
#     --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" \
#     --report_to="wandb" \
#     --learning_rate=1.41e-5 \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=8 \
#     --story_in_use="summary" \
#     --label_in_use="empathy" \
#     --output_dir="../sft_llama3-emp_2gpus-summary-2" \
#     --logging_steps=1 \
#     --num_train_epochs=8 \
#     --max_steps=-1 \
#     --torch_dtype='bfloat16' \
#     --use_peft \
#     --lora_r=64 \
#     --lora_alpha=16 \
#     --dataset_text_field="text" \
#     --datapath ../empathic-stories/data/


    # --load_in_8bit=True \

# accelerate launch --num_processes=2 --config_file=../deepspeed_zero3.yaml ../src/sft_llama3.py \
#     --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" \
#     --report_to="wandb" \
#     --run_name="fullpsft_llama3-emp_2gpus-summary" \
#     --learning_rate=1.41e-5 \
#     --per_device_train_batch_size=1 \
#     --gradient_accumulation_steps=64 \
#     --story_in_use="summary" \
#     --label_in_use="empathy" \
#     --output_dir="../fullpsft_llama3-emp_2gpus-summary" \
#     --logging_steps=1 \
#     --num_train_epochs=1 \
#     --max_steps=-1 \
#     --torch_dtype='bfloat16' \
#     --bf16_full_eval=True \
#     --bf16 \
#     --dataset_text_field="text" \
#     --datapath ../empathic-stories/data/

accelerate launch --num_processes=2 --config_file=../deepspeed_zero3.yaml ../src/sft_llama3.py \
    --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" \
    --report_to="wandb" \
    --run_name="fullpsft_llama3-emp_2gpus-summary-full_text_loss" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=64 \
    --story_in_use="summary" \
    --label_in_use="empathy" \
    --collator_type="full_text" \
    --output_dir="../fullpsft_llama3-emp_2gpus-summary-full_text_loss" \
    --logging_steps=1 \
    --num_train_epochs=2 \
    --max_steps=-1 \
    --torch_dtype='bfloat16' \
    --bf16_full_eval=True \
    --bf16 \
    --dataset_text_field="text" \
    --datapath ../empathic-stories/data/


# accelerate launch --num_processes=2 ../src/sft_llama3.py \
#     --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" \
#     --report_to="wandb" \
#     --run_name="sft_llama3-emp_2gpus-summary-3" \
#     --learning_rate=1.41e-5 \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=8 \
#     --story_in_use="summary" \
#     --label_in_use="empathy" \
#     --output_dir="../sft_llama3-emp_2gpus-summary-3" \
#     --logging_steps=1 \
#     --num_train_epochs=4 \
#     --max_steps=-1 \
#     --torch_dtype='bfloat16' \
#     --use_peft \
#     --lora_r=64 \
#     --lora_alpha=16 \
#     --dataset_text_field="text" \
#     --datapath ../empathic-stories/data/


# accelerate launch --num_processes=2 ../src/sft_llama3.py \
#     --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" \
#     --report_to="wandb" \
#     --run_name="sft_llama3-emp_2gpus-full-3" \
#     --learning_rate=1.41e-5 \
#     --per_device_train_batch_size=4 \
#     --gradient_accumulation_steps=8 \
#     --story_in_use="full" \
#     --label_in_use="empathy" \
#     --output_dir="../sft_llama3-emp_2gpus-full-3" \
#     --logging_steps=1 \
#     --num_train_epochs=4 \
#     --max_steps=-1 \
#     --torch_dtype='bfloat16' \
#     --use_peft \
#     --lora_r=64 \
#     --lora_alpha=16 \
#     --dataset_text_field="text" \
#     --datapath ../empathic-stories/data/