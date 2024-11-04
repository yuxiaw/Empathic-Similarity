
accelerate launch --num_processes=2 ../src/sft_llama3_cot.py \
    --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" \
    --report_to="wandb" \
    --run_name="sft_llama3-emp_2gpus-summary-cot-3" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --story_in_use="summary" \
    --label_in_use="empathy" \
    --output_dir="../sft_llama3-emp_2gpus-summary-cot-3" \
    --logging_steps=1 \
    --num_train_epochs=16 \
    --max_steps=-1 \
    --torch_dtype='bfloat16' \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16 \
    --dataset_text_field="text" \
    --datapath ../empathic-stories/data/ \
    --analysis_path ../data/

