
pip install transformers==4.21.2 datasets==1.18.0 librosa jiwer evaluate numpy==1.22

python run_speech_recognition_ctc.py \
	--dataset_name="common_voice" \
	--model_name_or_path="fav-kky/wav2vec2-base-cs-80k-ClTRUS" \
	--dataset_config_name="cs" \
	--output_dir="/storage/plzen4-ntis/home/sulc.././wav2vec2-common_voice-cs" \
	--overwrite_output_dir \
	--max_steps="10000" \
	--gradient_accumulation_steps="2" \
	--per_device_train_batch_size="4" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="sentence" \
	--length_column_name="input_length" \
	--save_steps="500" \
	--eval_steps="500" \
	--logging_steps="10" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--do_train \
	--do_eval
