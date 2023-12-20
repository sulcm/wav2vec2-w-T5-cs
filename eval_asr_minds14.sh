python eval.py \
	--model_name_or_path="/home/sulcm/models/wav2vec2/wav2vec2-cs-$1" \
    --output_dir="/home/sulcm/models/wav2vec2/eval_wav2vec2/wav2vec2-cs-$1/" \
	--dataset_name="PolyAI/minds14" \
	--dataset_config_name="cs-CZ" \
	--text_column_name="transcription" \
	--audio_column_name="audio" \
	--test_split_name="train" \
	--metrics "wer cer" \
	--chars_to_ignore ", ? . ! - \; \: \" “ % ‘ ” �"