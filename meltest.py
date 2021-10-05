import torch
import torchaudio
import pdb
import torchaudio.compliance.kaldi as kaldi
import math
def calc_mean_invstddev(feature):
    if len(feature.size()) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = feature.mean(0)
    var = feature.var(0)
    # avoid division by ~zero
    eps = 1e-8
    if (var < eps).any():
        return mean, 1.0 / (torch.sqrt(var) + eps)
    return mean, 1.0 / torch.sqrt(var)


def apply_mv_norm(features):
    mean, invstddev = calc_mean_invstddev(features)
    res = (features - mean) * invstddev
    return res


sound,fs = torchaudio.load("input0.wav")

window=torch.hann_window(window_length=400, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
sound_complex_R_I = torchaudio.functional.spectrogram(waveform=sound, pad=0, window=window, n_fft=400, hop_length=160, win_length=400, power=None, normalized=False)
sound_mag = torchaudio.functional.magphase(sound_complex_R_I)[0][0]



kk=torchaudio.transforms.MelScale(n_mels=90)(torch.log(sound_mag+1e-10)).permute(1,0)
kk1=torchaudio.transforms.MelScale(n_mels=90)(torch.log(sound_mag**2+1e-10)).permute(1,0)
fb = kaldi.fbank(
            sound,
            num_mel_bins=90,
            frame_length=25.0,
            frame_shift=10.0,
            window_type = 'hanning',
        )
pdb.set_trace()
kk2 = apply_mv_norm(kk)
kk22 = apply_mv_norm(kk1)
fb2 = apply_mv_norm(fb)
print("hi")