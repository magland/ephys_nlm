#!/usr/bin/env python

import os
import json
import hither
import kachery as ka
import numpy as np
from ephys_nlm import ephys_nlm_v1, ephys_nlm_v1_opts

def main():
    os.environ['HITHER_USE_SINGULARITY'] = 'TRUE'

    # recording_path = 'sha1://961f4a641af64dded4821610189f808f0192de4d/SYNTH_MEAREC_TETRODE/synth_mearec_tetrode_noise10_K10_C4/002_synth.json'
    # sorting_true_path = 'sha1://cce42806bcfe86f4f58c51aefb61f2c28a99f6cf/SYNTH_MEAREC_TETRODE/synth_mearec_tetrode_noise10_K10_C4/002_synth.firings_true.json'
    recording_path = 'sha1dir://fb52d510d2543634e247e0d2d1d4390be9ed9e20.synth_magland/datasets_noise20_K20_C4/001_synth'
    sorting_true_path = 'sha1dir://fb52d510d2543634e247e0d2d1d4390be9ed9e20.synth_magland/datasets_noise20_K20_C4/001_synth/firings_true.mda'
    # sorter_name = 'mountainsort4'
    sorter_name = 'ironclust'
    # sorter_name = 'kilosort2'

    test_sort(
        sorter_name=sorter_name,
        recording_path=recording_path,
        sorting_true_path=sorting_true_path
    )

    recording_denoised_path = denoise_recording(recording_path)
    test_sort(
        sorter_name=sorter_name,
        recording_path=recording_denoised_path,
        sorting_true_path=sorting_true_path
    )

def denoise_recording(recording_path):
    from spikeforest2_utils import AutoRecordingExtractor
    import spikeextractors as se

    R = AutoRecordingExtractor(recording_path)

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
    R_denoised, runtime_info = ephys_nlm_v1(recording=R, opts=opts, device=None, verbose=2)

    R_denoised_path = register_recording(recording=R_denoised, label='denoised', to=None)
    return R_denoised_path

def register_recording(*, recording, label, to):
    from spikeforest2_utils import MdaRecordingExtractor
    with ka.config(to=to):
        with hither.TemporaryDirectory() as tmpdir:
            recdir = tmpdir + '/recording'
            MdaRecordingExtractor.write_recording(recording=recording, save_path=recdir)
            raw_path = ka.store_file(recdir + '/raw.mda')
            obj = dict(
                raw=raw_path,
                params=ka.load_object(recdir + '/params.json'),
                geom=np.genfromtxt(ka.load_file(recdir + '/geom.csv'), delimiter=',').tolist()
            )
            obj['self_reference'] = ka.store_object(obj, basename='{}.json'.format(label))
            return ka.store_object(obj, basename='{}.json'.format(label))

def get_geom_from_recording(recording):
    channel_ids = recording.get_channel_ids()
    M = len(channel_ids)
    location0 = recording.get_channel_property(channel_ids[0], 'location')
    nd = len(location0)
    geom = np.zeros((M, nd))
    for ii in range(len(channel_ids)):
        location_ii = recording.get_channel_property(channel_ids[ii], 'location')
        geom[ii, :] = list(location_ii)
    return geom

def test_sort(
    sorter_name,
    recording_path,
    sorting_true_path,
    num_jobs=1,
    job_handler=None,
    container='default'
):
    from spikeforest2 import sorters
    from spikeforest2 import processing
    import kachery as ka

    # for now, in this test, don't use gpu for irc
    gpu = sorter_name in ['kilosort2', 'kilosort', 'tridesclous']
    cache = 'default_readwrite'

    sorting_results = []
    with ka.config(fr='default_readonly'):
        with hither.config(container=container, gpu=gpu, job_handler=job_handler, cache=cache), hither.job_queue():
            sorter = getattr(sorters, sorter_name)
            for _ in range(num_jobs):
                sorting_result = sorter.run(
                    recording_path=recording_path,
                    sorting_out=hither.File()
                )
                sorting_results.append(sorting_result)

    assert sorting_result.success

    sorting_result = sorting_results[0]
    with ka.config(fr='default_readonly'):
        with hither.config(container='default', gpu=False):
            compare_result = processing.compare_with_truth.run(
                sorting_path=sorting_result.outputs.sorting_out,
                sorting_true_path=sorting_true_path,
                json_out=hither.File()
            )

    assert compare_result.success

    obj = ka.load_object(compare_result.outputs.json_out._path)

    aa = _average_accuracy(obj)

    print(F'AVERAGE-ACCURACY: {aa}')

    print('Passed.')

def _average_accuracy(obj):
    accuracies = [float(obj[i]['accuracy']) for i in obj.keys()]
    print(accuracies)
    return np.mean(accuracies)

if __name__ == '__main__':
    main()