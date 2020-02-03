import os
import numpy as np
import pytest
from ephys_nlm import ephys_nlm_v1, ephys_nlm_v1_opts
from ephys_nlm import example_datasets

import spikeextractors as se

def main():
    test_accuracy_of_denoising()
    test_params()
    test_coverage()

@pytest.mark.accuracy
def test_accuracy_of_denoising():
    # Test the accuracy of denoising
    duration=10
    num_channels=4
    sampling_frequency=30000
    K=10
    seed=None

    upsamplefac = 13

    waveforms, geom = example_datasets.synthesize_random_waveforms(K=K, M=num_channels, average_peak_amplitude=-100,
                                                  upsamplefac=upsamplefac, seed=seed)
    times, labels = example_datasets.synthesize_random_firings(
        K=K, duration=duration, sampling_frequency=sampling_frequency, seed=seed)
    labels = labels.astype(np.int64)
    SX = se.NumpySortingExtractor()
    SX.set_times_labels(times, labels)

    SX.set_sampling_frequency(sampling_frequency)

    recordings = []
    for noise_level in [0, 10]:
        X = example_datasets.synthesize_timeseries(
            sorting=SX,
            waveforms=waveforms,
            noise_level=noise_level,
            sampling_frequency=sampling_frequency,
            duration=duration,
            waveform_upsamplefac=upsamplefac,
            seed=seed
        )

        RX = se.NumpyRecordingExtractor(
            timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        
        recordings.append(RX)
    
    recording_without_noise = recordings[0]
    recording_with_noise = recordings[1]
    
    opts = ephys_nlm_v1_opts(
        multi_neighborhood=False,
        block_size_sec=30,
        clip_size=30,
        sigma='auto',
        sigma_scale_factor=1,
        whitening='auto',
        whitening_pctvar=90,
        denom_threshold=30
    )

    recording_denoised, runtime_info = ephys_nlm_v1(
        recording=recording_with_noise,
        opts=opts,
        device=None, # detect from the EPHYS_NLM_DEVICE environment variable
        verbose=2
    )

    traces_with_noise = recording_with_noise.get_traces()
    traces_without_noise = recording_without_noise.get_traces()
    traces_denoised = recording_denoised.get_traces()

    std_noise_before = np.sqrt(np.var(traces_without_noise - traces_with_noise))
    std_noise_after = np.sqrt(np.var(traces_without_noise - traces_denoised))
    print(f'std_noise_before = {std_noise_before}; std_noise_after = {std_noise_after};')

    assert std_noise_after < 0.3 * std_noise_before

pytest.mark.params
def test_params():
    # Test different parameter combinations
    recording, sorting_true = example_datasets.toy_example(duration=3, num_channels=4, K=20, seed=4)

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
        block_size_sec=1.5,
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

    for opts in [opts1, opts2, opts3]:
        recording_denoised, runtime_info = ephys_nlm_v1(
            recording=recording,
            opts=opts,
            device=None, # detect from the EPHYS_NLM_DEVICE environment variable
            verbose=2
        )
        a = np.sum(recording_denoised.get_traces())
        print(a)
    
def test_coverage():
    # Some quick tests to increase code coverage

    recording, sorting_true = example_datasets.toy_example(duration=3, num_channels=4, K=20, seed=4)
    
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

    for opts in [opts1]:
        recording_denoised, runtime_info = ephys_nlm_v1(
            recording=recording,
            opts=opts,
            device='cpu',
            verbose=5
        )
        print('sampling freq. of output:', recording_denoised.get_sampling_frequency())

    # Make sure we get an exception in the following cases: (code coverage)
    optsA = ephys_nlm_v1_opts(
        multi_neighborhood=False, clip_size=30, sigma='auto', sigma_scale_factor=1, whitening='auto', whitening_pctvar=90, denom_threshold=100,
        block_size_sec=None
    )

    for opts in [optsA]:
        with pytest.raises(Exception):
            recording_denoised, runtime_info = ephys_nlm_v1(
                recording=recording,
                opts=opts,
                device=None, # detect from the EPHYS_NLM_DEVICE environment variable
                verbose=2
            )


if __name__ == '__main__':
    main()