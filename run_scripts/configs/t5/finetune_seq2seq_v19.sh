# export WANDB_DISABLED=true

python run_seq2seq.py \
	--model_name_or_path="/storage/plzen4-ntis/projects/public/Lehecka/t5_32k_cccs_jmzw.v2" \
	--output_dir="/storage/plzen4-ntis/home/sulcm01/outputs/t5/t5-spellchecker-cs-v19/" \
	--overwrite_output_dir \
	--dataset_name="/storage/plzen4-ntis/home/sulcm01/datasets/t5/asr-correction-cs-v23" \
	--source="asr_transcription" \
	--target="target_output" \
	--eval_metrics wer cer \
	--source_prefix="spell check: " \
	--evaluation_strategy="steps" \
	--max_steps="20000" \
	--learning_rate="3e-4" \
	--warmup_steps="1000" \
	--max_source_length="1024" \
	--max_target_length="128" \
	--num_beams="4" \
	--generation_num_beams="4" \
	--generation_max_length="64" \
	--sortish_sampler \
	--predict_with_generate \
	--custom_generation_config="./configs/t5/generation_config/t5_v1.json" \
	--do_train \
	--do_eval \
	--load_best_model_at_end=True \
	--metric_for_best_model="wer" \
	--greater_is_better=False \
	--per_device_train_batch_size="64" \
	--gradient_accumulation_steps="2" \
	--save_steps="1000" \
	--eval_steps="1000" \
	--logging_steps="100" \
	--save_total_limit="2" \
	--dataloader_num_workers="8" \
	--preprocessing_num_workers="8"
