python model_eval.py \
	--model_name_or_path="/home/sulcm/models/wav2vec2/wav2vec2-cs-$1" \
    --output_dir="./results/wav2vec2/wav2vec2-cs-$1/" \
	--dataset_name="facebook/voxpopuli" \
	--dataset_config_name="cs" \
	--text_column_name="normalized_text" \
	--audio_column_name="audio" \
	--test_split_name="test" \
	--metrics "wer cer" \
	--chars_to_ignore ", ? . ! - \; \: \" “ % ‘ ” �"