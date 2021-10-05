# ./yh_transformer_norm_ce.sh 1

# sed -i 's/shortword/sentence_pretrain_trainval/g' ./yh_transformer_norm_ce.sh
# sed -i 's/MAX_EPOCH=10/MAX_EPOCH=11/g' ./yh_transformer_norm_ce.sh
# sed -i 's/LR_SCHEDULER=reduce_lr_on_plateau/LR_SCHEDULER=fixed/g' ./yh_transformer_norm_ce.sh
# ./yh_transformer_norm_ce.sh 1

# sed -i 's/MAX_EPOCH=11/MAX_EPOCH=30/g' ./yh_transformer_norm_ce.sh
# sed -i 's/LR_SCHEDULER=fixed/LR_SCHEDULER=reduce_lr_on_plateau/g' ./yh_transformer_norm_ce.sh
# ./yh_transformer_norm_ce.sh 1


# sed -i 's/sentence_pretrain_trainval/sentence_noisy_low/g' ./yh_transformer_norm_ce.sh
# sed -i 's/MAX_EPOCH=30/MAX_EPOCH=35/g' ./yh_transformer_norm_ce.sh
# sed -i 's/LR_SCHEDULER=reduce_lr_on_plateau/LR_SCHEDULER=fixed/g' ./yh_transformer_norm_ce.sh
# sed -i 's/LR=1/LR=0.1/g' ./yh_transformer_norm_ce.sh
./yh_transformer_norm_ce.sh 1

sed -i 's/new_sentence_noisy_low/new_sentence_noisy_eq/g' ./yh_transformer_norm_ce.sh
sed -i 's/MAX_EPOCH=35/MAX_EPOCH=75/g' ./yh_transformer_norm_ce.sh
./yh_transformer_norm_ce.sh 1
