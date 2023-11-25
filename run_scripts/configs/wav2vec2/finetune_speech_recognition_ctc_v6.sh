python run_speech_recognition_ctc.py \
	--model_name_or_path="/storage/plzen4-ntis/projects/public/Lehecka/wav2vec2-base-cs-80k" \
	--output_dir="/storage/plzen4-ntis/home/sulcm01/outputs/wav2vec2/wav2vec2-cs-v6/" \
	--overwrite_output_dir \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="cs" \
	--text_column_name="sentence" \
	--audio_column_name="audio" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--eval_metrics wer cer \
	--evaluation_strategy="steps" \
	--max_steps="20000" \
	--learning_rate="5e-5" \
	--warmup_steps="1000" \
	--layerdrop="0.05" \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--freeze_feature_encoder \
	--activation_dropout="0.1" \
	--mask_time_prob="0.3" \
	--mask_time_length="10" \
	--mask_feature_prob="0.1" \
	--mask_feature_length="64" \
	--fp16 \
	--group_by_length \
	--do_train \
	--do_eval \
	--load_best_model_at_end=True \
	--metric_for_best_model="wer" \
	--greater_is_better=False \
	--gradient_accumulation_steps="8" \
	--per_device_train_batch_size="16" \
	--save_steps="500" \
	--eval_steps="500" \
	--logging_steps="10" \
	--save_total_limit="2" \
	--gradient_checkpointing
