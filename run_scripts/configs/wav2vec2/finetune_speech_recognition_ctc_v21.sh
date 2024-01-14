python run_speech_recognition_ctc.py \
	--model_name_or_path="/storage/plzen4-ntis/projects/public/Lehecka/wav2vec2-base-cs-80k" \
	--output_dir="/storage/plzen4-ntis/home/sulcm01/outputs/wav2vec2/wav2vec2-cs-v21/" \
	--overwrite_output_dir \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="cs" \
	--text_column_name="sentence" \
	--audio_column_name="audio" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--eval_metrics wer cer \
	--evaluation_strategy="steps" \
	--max_steps="40000" \
	--learning_rate="9e-6" \
	--warmup_steps="4000" \
	--layerdrop="0.1" \
	--activation_dropout="0.1" \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--freeze_feature_encoder \
	--mask_time_prob="0.4" \
	--mask_time_length="5" \
	--mask_feature_prob="0.1" \
	--mask_feature_length="64" \
	--fp16 \
	--group_by_length \
	--length_column_name="input_length" \
	--do_train \
	--do_eval \
	--load_best_model_at_end=True \
	--metric_for_best_model="wer" \
	--greater_is_better=False \
	--gradient_accumulation_steps="2" \
	--per_device_train_batch_size="128" \
	--save_steps="2000" \
	--eval_steps="2000" \
	--logging_steps="100" \
	--save_total_limit="2" \
	--gradient_checkpointing \
	--dataloader_num_workers="8" \
	--preprocessing_num_workers="8"
