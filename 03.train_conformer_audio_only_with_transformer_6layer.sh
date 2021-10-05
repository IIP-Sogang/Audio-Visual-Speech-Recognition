./jw_conformer_audio_only_with_transformer_6layer.sh 1

# sed -i 's/shortword/sentence_pretrain_trainval/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# sed -i 's/MAX_EPOCH=10/MAX_EPOCH=11/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# sed -i 's/LR_SCHEDULER=reduce_lr_on_plateau/LR_SCHEDULER=fixed/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# ./jw_conformer_audio_only_with_transformer_6layer.sh 1

# sed -i 's/MAX_EPOCH=11/MAX_EPOCH=30/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# sed -i 's/LR_SCHEDULER=fixed/LR_SCHEDULER=reduce_lr_on_plateau/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# ./jw_conformer_audio_only_with_transformer_6layer.sh 1


# sed -i 's/sentence_pretrain_trainval/sentence_noisy_low/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# sed -i 's/MAX_EPOCH=30/MAX_EPOCH=35/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# sed -i 's/LR_SCHEDULER=reduce_lr_on_plateau/LR_SCHEDULER=fixed/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# sed -i 's/LR=1/LR=0.1/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# ./jw_conformer_audio_only_with_transformer_6layer.sh 1

# sed -i 's/sentence_noisy_low/sentence_noisy_eq/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# sed -i 's/MAX_EPOCH=35/MAX_EPOCH=75/g' ./jw_conformer_audio_only_with_transformer_6layer.sh
# ./jw_conformer_audio_only_with_transformer_6layer.sh 1



# sed -i 's/MAX_EPOCH=75/MAX_EPOCH=150/g' ./jw_conformer_audio_only_no_position.sh
# ./jw_conformer_audio_only_no_position.sh 1