python create_dataset_4_t5.py \
	--model_name_or_path="/home/sulcm/models/wav2vec2/wav2vec2-cs-$1" \
    --output_dir="/home/sulcm/datasets/t5/asr-correction-cs-$1" \
	--dataset_name="facebook/voxpopuli" \
	--dataset_config_name="cs" \
	--text_column_name="normalized_text" \
	--audio_column_name="audio" \
	--split_names="train validation test" \
	--metrics "wer cer" \
	--chars_to_ignore ", ? . ! - \; \: \" “ % ‘ ” �"