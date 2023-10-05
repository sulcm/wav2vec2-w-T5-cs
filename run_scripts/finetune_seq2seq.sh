# python run_seq2seq.py \
# 	--model_name_or_path="" \
# 	--output_dir="/storage/plzen4-ntis/home/sulcm01/outputs//" \
# 	--overwrite_output_dir \
# 	--dataset_name="" \
# 	--dataset_config_name="" \
# 	--text_column_name="" \
# 	--audio_column_name="" \
# 	--train_split_name="train+validation" \
# 	--eval_split_name="test" \
# 	--length_column_name="input_length" \
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
# 	--gradient_accumulation_steps="2" \
# 	--per_device_train_batch_size="4" \
# 	--save_steps="500" \
# 	--eval_steps="500" \
# 	--logging_steps="10" \
# 	--save_total_limit="2" \
# 	--gradient_checkpointing