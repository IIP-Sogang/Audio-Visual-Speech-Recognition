# ./jw_conformer_audio_only.sh 1

# sed -i 's/shortword/sentence_pretrain_trainval/g' ./jw_conformer_audio_only.sh
# sed -i 's/MAX_EPOCH=10/MAX_EPOCH=11/g' ./jw_conformer_audio_only.sh
# sed -i 's/LR_SCHEDULER=reduce_lr_on_plateau/LR_SCHEDULER=fixed/g' ./jw_conformer_audio_only.sh
# ./jw_conformer_audio_only.sh 1

# sed -i 's/MAX_EPOCH=11/MAX_EPOCH=30/g' ./jw_conformer_audio_only.sh
# sed -i 's/LR_SCHEDULER=fixed/LR_SCHEDULER=reduce_lr_on_plateau/g' ./jw_conformer_audio_only.sh
./jw_avse_avsr_transformer.sh 1


sed -i 's/sentence_pretrain_trainval/new_sentence_noisy_low/g' ./jw_avse_avsr_transformer.sh
sed -i 's/MAX_EPOCH=20/MAX_EPOCH=25/g' ./jw_avse_avsr_transformer.sh
sed -i 's/LR_SCHEDULER=reduce_lr_on_plateau/LR_SCHEDULER=fixed/g' ./jw_avse_avsr_transformer.sh
sed -i 's/LR=1/LR=0.1/g' ./jw_avse_avsr_transformer.sh
./jw_avse_avsr_transformer.sh 1

sed -i 's/new_sentence_noisy_low/new_sentence_noisy_eq/g' ./jw_avse_avsr_transformer.sh
sed -i 's/MAX_EPOCH=25/MAX_EPOCH=65/g' ./jw_avse_avsr_transformer.sh
./jw_avse_avsr_transformer.sh 1
