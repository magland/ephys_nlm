import numpy as np
from ephys_nlm import ephys_nlm_v1, ephys_nlm_v1_opts

import spikeextractors as se

def main():
    test_1()

def test_1():
    recording, sorting_true = se.example_datasets.toy_example(duration=3, num_channels=4, K=20, seed=4)

    opts1 = ephys_nlm_v1_opts(
        multi_neighborhood=False,
        block_size_sec=1,
        clip_size=30,
        sigma='auto',
        sigma_scale_factor=1,
        whitening='auto',
        whitening_pctvar=90,
        denom_threshold=30
    )

    opts2 = ephys_nlm_v1_opts(
        multi_neighborhood=True,
        neighborhood_adjacency_radius=1.5,
        block_size_sec=1,
        clip_size=30,
        sigma='auto',
        sigma_scale_factor=1,
        whitening='auto',
        whitening_pctvar=90,
        denom_threshold=30
    )

    opts3 = ephys_nlm_v1_opts(
        multi_neighborhood=False,
        block_size_sec=30,
        clip_size=30,
        sigma='auto',
        sigma_scale_factor=1,
        whitening='auto',
        whitening_pctvar=90,
        denom_threshold=100
    )

    for opts in [opts1]:
        recording_denoised, runtime_info = ephys_nlm_v1(
            recording=recording,
            opts=opts,
            device='cpu', # cuda is recommended for non-demo situations
            verbose=2
        )
        a = np.sum(recording_denoised.get_traces())

if __name__ == '__main__':
    main()