python run_speech_recognition_ctc.py \
	--model_name_or_path="/storage/plzen4-ntis/projects/public/Lehecka/wav2vec2-base-cs-80k" \
	--output_dir="/storage/plzen4-ntis/home/sulcm01/outputs//" \
	--overwrite_output_dir \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="cs" \
	--text_column_name="sentence" \
	--audio_column_name="audio" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--length_column_name="input_length" \
	--eval_metrics="wer" \
	--evaluation_strategy="steps" \
	--max_steps="10000" \
	--learning_rate="1e-4" \
	--weight_decay="0.005" \
	--warmup_steps="500" \
	--layerdrop="0.0" \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--freeze_feature_encoder \
	--fp16 \
	--group_by_length \
	--do_train \
	--do_eval \
	--gradient_accumulation_steps="2" \
	--per_device_train_batch_size="8" \
	--save_steps="500" \
	--eval_steps="100" \
	--logging_steps="10" \
	--save_total_limit="2" \
	--gradient_checkpointing