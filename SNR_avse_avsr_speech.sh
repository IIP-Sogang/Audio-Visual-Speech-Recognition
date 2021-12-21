# alias
sclite='/home/nas3/user/jungwook/SCTK/src/sclite/sclite'

#Inference parmas
IDX_GPU=$1
ARCH=$2
CODE=$3
SET=$4
EPOCH=$5
VID_OFS=$8
GAUSSIAN_VARIANCE=$9
GAUSSIAN_POW=$10

TASK=audio_visual_se_sr
DIR_FOR_PREPROCESSED_DATA=$6
MODEL_PATH=/home/nas/user/jungwook/DCM_vgg_transformer/result/$TASK/$ARCH/model/$CODE
MAX_TOKEN=6000
N_BEST=1
BEAM=35
RES_DIR=/home/nas/user/jungwook/DCM_vgg_transformer/result_jw/$TASK/$ARCH/prediction/$CODE
WORD_RES_REPORT=/home/nas/user/jungwook/DCM_vgg_transformer/result_jw/$TASK/$ARCH/{$7}_SNR_TEST/$CODE/$EPOCH/$SET/WER_$CODE
UNIT_RES_REPORT=/home/nas/user/jungwook/DCM_vgg_transformer/result_jw/$TASK/$ARCH/{$7}_SNR_TEST/$CODE/$EPOCH/$SET/CER_$CODE
CHECK_POINT=checkpoint$EPOCH
USER_DIR=/home/nas/user/jungwook/DCM_vgg_transformer/examples/$TASK

if [ -f "$WORD_RES_REPORT" ]; then
	exit 0
fi

mkdir -p /home/nas/user/jungwook/DCM_vgg_transformer/result_jw/$TASK/$ARCH/{$7}_SNR_TEST/$CODE/$EPOCH/$SET

CUDA_VISIBLE_DEVICES=$IDX_GPU python3.7 examples/$TASK/infer.py \
	$DIR_FOR_PREPROCESSED_DATA \
	--task $TASK \
	--max-tokens $MAX_TOKEN \
	--nbest $N_BEST \
	--path $MODEL_PATH/$CHECK_POINT.pt \
	--beam $BEAM \
	--results-path $RES_DIR \
	--gen-subset $SET \
	--user-dir $USER_DIR \
	--vid_ofs $VID_OFS \
	--dataset-method 2 \
	#--gaussian_variance $GAUSSIAN_VARIANCE \
	#--gaussian_pow_value $GAUSSIAN_POW \

#Compute WER
#$sclite -r ${RES_DIR}/ref.word-$CHECK_POINT.pt-${SET}.txt -h ${RES_DIR}/hypo.word-$CHECK_POINT.pt-${SET}.txt -i rm -o all stdout > $WORD_RES_REPORT
#$sclite -r ${RES_DIR}/ref.units-$CHECK_POINT.pt-${SET}.txt -h ${RES_DIR}/hypo.units-$CHECK_POINT.pt-${SET}.txt -i rm -o all stdout > $UNIT_RES_REPORT
