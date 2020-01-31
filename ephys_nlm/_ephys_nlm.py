import spikeextractors as se
import numpy as np

class EphysNlmOpts:
    def __init__(self, *,
            clip_size=30,
            device='cpu'
        ):
        self.clip_size = clip_size
        self.device = device

def ephys_nlm_opts():
    opts = EphysNlmOpts()
    return opts

def ephys_nlm(recording: se.RecordingExtractor, *, opts: EphysNlmOpts) -> se.RecordingExtractor:
    channel_ids = recording.get_channel_ids()
    M = len(channel_ids)
    N = recording.get_num_frames()
    T = opts.clip_size
    device = opts.device
    assert T % 2 == 0, 'clip size must be even.'
    T2 = int(T/2)

    if device == 'cuda':
        print('Using device=cuda')
    elif device == 'cpu':
        print('Using device=cpu. Warning: GPU is much faster. Recommendation: device=cuda')
    else:
        raise Exception(f'Invalid device: {device}')

    traces_denoised = np.zeros((M, N), dtype=np.float32)

    return recording
