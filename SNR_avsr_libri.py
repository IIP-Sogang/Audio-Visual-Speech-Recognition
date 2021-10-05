import subprocess
import argparse
import sys

parser = argparse.ArgumentParser(description='transformer AVSR evalution code')

parser.add_argument('--gpu', type=str, default='2', help='gpu index')
parser.add_argument('--arch', type=str, required=True, help='model architecture')
parser.add_argument('--code', type=str, required=True, help='model code')
parser.add_argument('--start', type=int, default=34, help='start epoch')
parser.add_argument('--end', type=int, default=34, help='end epoch')
parser.add_argument('--data', type=str, default='libri', help='LRS2-BBC, LRS3-TED and both ot them')
parser.add_argument('--words', type=int, default=0, help='#words in sentence inference')
parser.add_argument('--vid_off', type=int, default=0, help='#frames in video offset')
#parser.add_argument('--gaussian_variance', type=int, default=5, help='variance value of the gaussian function')
#parser.add_argument('--gaussian_pow_value', type=float, default=1.0, help='the value of gaussian pow value')

args = parser.parse_args()

IDX_GPU=args.gpu
ARCH=args.arch
CODE=args.code
SNR=['clean']
st=args.start
ed=args.end
db = '/home/nas/DB/[DB]_for_fairseq/[DB]_librispeech/preprocessed_data_character'


if st <= ed:
    EPOCH=range(st,ed+1)
elif st > ed:
    EPOCH=range(ed,st+1)
    EPOCH=EPOCH[::-1]
    SNR=SNR[::-1]

if args.vid_off == 1:
    SNR=['-12_offset','-10_offset','-8_offset','-6_offset','-4_offset','-2_offset','0_offset','2_offset','4_offset','6_offset','8_offset','10_offset','12_offset']
    db = '/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/preprocessed_data/character/sentence_vid_ofs'

    for epoch in EPOCH:
        for snr in SNR:
            offsets = int(snr.split('_')[0])
            command = "sh ./SNR_avsr.sh " + IDX_GPU + " " + ARCH + " " + CODE + " " + "test_" + str(snr) + " " + str(epoch) + " " + str(db) + " " + str(args.data) + " " + str(offsets)
            print(command)
            subprocess.call(command, shell=True)
    sys.exit()

elif args.vid_off == 2:
    SNR=['-12_offset','-10_offset','-8_offset','-6_offset','-4_offset','-2_offset','0_offset','2_offset','4_offset','6_offset','8_offset','10_offset','12_offset']
    db = '/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/preprocessed_data/character/sentence_vid_ofs_0dB'

    for epoch in EPOCH:
        for snr in SNR:
            offsets = int(snr.split('_')[0])
            command = "sh ./SNR_avsr.sh " + IDX_GPU + " " + ARCH + " " + CODE + " " + "test_" + str(snr) + " " + str(epoch) + " " + str(db) + " " + str(args.data) + " " + str(offsets)
            print(command)
            subprocess.call(command, shell=True)
    sys.exit()

elif args.vid_off == 3:
    SNR=['-12_offset','-10_offset','-8_offset','-6_offset','-4_offset','-2_offset','0_offset','2_offset','4_offset','6_offset','8_offset','10_offset','12_offset']
    db = '/home/nas/DB/[DB]_for_fairseq/[DB]_LRS_con/preprocessed_data/character/sentence_vid_ofs_m5dB'

    for epoch in EPOCH:
        for snr in SNR:
            offsets = int(snr.split('_')[0])
            command = "sh ./SNR_avsr.sh " + IDX_GPU + " " + ARCH + " " + CODE + " " + "test_" + str(snr) + " " + str(epoch) + " " + str(db) + " " + str(args.data) + " " + str(offsets)
            print(command)
            subprocess.call(command, shell=True)
    sys.exit()

for epoch in EPOCH:
    for snr in SNR:
        command = "sh ./SNR_avsr_libri.sh " + IDX_GPU + " " + ARCH + " " + CODE + " " + "test-" + str(snr) + " " + str(epoch) + " " + str(db) + " " + str(args.data) + " " + str(0)
        print(command)
        subprocess.call(command, shell=True)
