python model_eval.py \
	--w2v2_model_name_or_path="/home/sulcm/models/wav2vec2/wav2vec2-cs-$1" \
    --output_dir="./results/" \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="cs" \
	--text_column_name="sentence" \
	--audio_column_name="audio" \
	--test_split_name="test" \
	--metrics "wer cer" \
	--chars_to_ignore ", ? . ! - \; \: \" “ % ‘ ” �"