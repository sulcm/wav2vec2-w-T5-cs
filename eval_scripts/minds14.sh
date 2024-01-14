python model_eval.py \
	--model_name_or_path="/home/sulcm/models/wav2vec2/wav2vec2-cs-$1" \
    --output_dir="./results/wav2vec2/wav2vec2-cs-$1/" \
	--dataset_name="PolyAI/minds14" \
	--dataset_config_name="cs-CZ" \
	--text_column_name="transcription" \
	--audio_column_name="audio" \
	--test_split_name="train" \
	--metrics "wer cer" \
	--chars_to_ignore ", ? . ! - \; \: \" “ % ‘ ” �"