# python run_seq2seq.py \
# 	# Base model
# 	--model_name_or_path="/storage/plzen4-ntis/projects/public/Lehecka/wav2vec2-base-cs-80k" \
# 	# Replace * with dir
# 	--output_dir="/storage/plzen4-ntis/home/sulcm01/outputs/*" \
# 	--overwrite_output_dir \
# 	# Dataset args (match --dataset_config_name, --text_column_name, --audio_column_name)
# 	--dataset_name="mozilla-foundation/common_voice_11_0" \
# 	--dataset_config_name="cs" \
# 	--text_column_name="sentence" \
# 	--audio_column_name="audio" \
# 	--train_split_name="train+validation" \
# 	--eval_split_name="test" \
# 	--length_column_name="input_length" \
# 	# Trainig args
# 	--eval_metrics="wer" \
# 	--evaluation_strategy="steps" \
# 	--max_steps="10000" \
# 	--learning_rate="3e-4" \
# 	--warmup_steps="500" \
# 	--layerdrop="0.0" \
# 	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
# 	--freeze_feature_encoder \
# 	--fp16 \
# 	--group_by_length \
# 	--do_train \
# 	--do_eval \
# 	# Run args
# 	--gradient_accumulation_steps="2" \
# 	--per_device_train_batch_size="4" \
# 	--save_steps="500" \
# 	--eval_steps="500" \
# 	--logging_steps="10" \
# 	--save_total_limit="2" \
# 	--gradient_checkpointing