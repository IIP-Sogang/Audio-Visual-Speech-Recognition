# alias
sclite='/home/nas3/user/jungwook/SCTK/src/sclite/sclite'
# Select GPU
IDX_GPU=1
# Preparing dataset
DIR_TO_SAVE_RAW_DATA=/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/raw_data
DIR_FOR_PREPROCESSED_DATA=/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/preprocessed_data/character/sentence_noisy_eq 

# if [ $1 -eq 0 ]; then
# 	sh /home/nas/user/yong/fairseq/examples/audio_visual_speech_recognition/datasets/prepare-LRS.sh
# fi

# Training
TASK=audio_visual_speech_recognition
MAX_EPOCH=75
NUM_WORKERS=20
ARCH=BiModalvggconformer_avsr_base_512_6_6
CODE=VGG_base_wi_conformer_CE_256_sakje
MODEL_PATH=/home/nas/user/jungwook/DCM_vgg_transformer/result/$TASK/$ARCH/model/$CODE
LR=0.1
LR_SHRINK=0.5
# Available --lr-scheduler options 
# fixed, polynomial_decay, triangular, reduce_lr_on_plateau, inverse_sqrt, cosine #
LR_SCHEDULER=fixed
CLIP_NORM=10.0
MAX_TOKEN=6000
CRITERION=cross_entropy_acc
TENSORBOARD=/home/nas/user/jungwook/DCM_vgg_transformer/result/$TASK/$ARCH/tensorboard/$CODE
USER_DIR=/home/nas/user/jungwook/DCM_vgg_transformer/examples/$TASK/

if [ $1 -eq 1 ]; then
	CUDA_VISIBLE_DEVICES=$IDX_GPU python train_practice.py \
		$DIR_FOR_PREPROCESSED_DATA \
		--save-dir $MODEL_PATH \
		--max-epoch $MAX_EPOCH \
		--task $TASK \
		--arch $ARCH \
		--num-workers $NUM_WORKERS \
		--lr $LR \
		--lr-shrink $LR_SHRINK \
		--lr-scheduler $LR_SCHEDULER\
		--optimizer adadelta \
		--adadelta-eps 1e-8 \
		--adadelta-rho 0.95 \
		--clip-norm $CLIP_NORM  \
		--max-tokens $MAX_TOKEN \
		--log-format json \
		--log-interval 1 \
		--criterion $CRITERION \
		--tensorboard-logdir $TENSORBOARD \
		--user-dir $USER_DIR \
		--fp16
		#--no-save
	
fi

#Inference
# MAX_TOKEN=25000
# N_BEST=1
# BEAM=20
# BATCH_SIZE=10
# RES_DIR=/home/nas/user/jungwook/fairseq/result/$TASK/$ARCH/prediction
# RES_REPORT=/home/nas/user/jungwook/fairseq/result/$TASK/$ARCH/report
# SET=test
# CHECK_POINT=checkpoint_best

# if [ $1 -eq 2 ]; then
# 	CUDA_VISIBLE_DEVICES=$IDX_GPU python examples/$TASK/infer.py \
# 		$DIR_FOR_PREPROCESSED_DATA \
# 		--task $TASK \
# 		--max-tokens $MAX_TOKEN \
# 		--nbest $N_BEST \
# 		--path $MODEL_PATH/$CHECK_POINT.pt \
# 		--beam $BEAM \
# 		--results-path $RES_DIR \
# 		--batch-size $BATCH_SIZE \
# 		--gen-subset $SET \
# 		--user-dir $USER_DIR
	
# 	# Compute WER
# 	$sclite -r ${RES_DIR}/ref.word-$CHECK_POINT.pt-${SET}.txt -h ${RES_DIR}/hypo.word-$CHECK_POINT.pt-${SET}.txt -i rm -o all stdout > $RES_REPORT
# fi
