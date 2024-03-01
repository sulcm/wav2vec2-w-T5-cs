# export WANDB_DISABLED=true

python run_seq2seq.py \
	--model_name_or_path="/storage/plzen4-ntis/projects/public/Lehecka/t5_32k_cccs_jmzw.v2" \
	--output_dir="/storage/plzen4-ntis/home/sulcm01/outputs/t5/t5-spellchecker-cs-v13/" \
	--overwrite_output_dir \
	--dataset_name="/storage/plzen4-ntis/home/sulcm01/datasets/t5/asr-correction-cs-v23" \
	--eval_metrics wer cer \
	--source_prefix="spell check: " \
	--evaluation_strategy="steps" \
	--max_steps="20000" \
	--learning_rate="1e-3" \
	--warmup_steps="800" \
	--label_smoothing_factor="0.1" \
	--max_source_length="256" \
	--max_target_length="256" \
	--generation_num_beams="4" \
	--generation_max_length="64" \
	--sortish_sampler \
	--predict_with_generate \
	--do_train \
	--do_eval \
	--load_best_model_at_end=True \
	--metric_for_best_model="wer" \
	--greater_is_better=False \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="8" \
	--save_steps="1000" \
	--eval_steps="1000" \
	--logging_steps="100" \
	--save_total_limit="2" \
	--dataloader_num_workers="8" \
	--preprocessing_num_workers="8"
