[![Build Status](https://travis-ci.org/magland/ephys_nlm.svg?branch=master)](https://travis-ci.org/magland/ephys_nlm)
[![codecov](https://codecov.io/gh/magland/ephys_nlm/branch/master/graph/badge.svg)](https://codecov.io/gh/magland/ephys_nlm)

# ephys_nlm

Non-local means denoising of multi-channel electrophysiology timeseries using PyTorch.

The software is in the alpha stage of development. Testers and contributers are welcome to assist.

## Overview

Electrophysiology recordings contain spike waveforms
superimposed on a noisy background signal. Each detected neuron may fire
hundreds or even thousands of times over the duration of a recording,
producing a very similar voltage trace at each event. Here we aim to
leverage this redundancy in order to denoise the recording in a preprocessing step,
prior to spike sorting or other analyses.
Our approach is to use non-local means to suppress the noisy signal while
retaining the part of the signal that repeats.

Non-local means is a denoising algorithm that has primarily been used for 2d
images, but it can also be adapted for use with 1d signals. Let <img src="/tex/41703bc15dc133f4d6c45dd947a73828.svg?invert_in_darkmode&sanitize=true" align=middle width=28.594206299999986pt height=14.15524440000002pt/> be
the multi-channel voltage trace for <img src="/tex/e000d6acc762ceb268274567a298c7d5.svg?invert_in_darkmode&sanitize=true" align=middle width=90.62011694999998pt height=22.465723500000017pt/> and <img src="/tex/c2ff91972b6fc450be3647c0cc19df56.svg?invert_in_darkmode&sanitize=true" align=middle width=79.38336614999999pt height=22.465723500000017pt/> where <img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/> is
the number of electrode channels, and <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> is the number of timepoints. If
computation time were not a concern, we would ideally denoise each half-overlapping clip (or snippet) <img src="/tex/9f7365802167fff585175c1750674d42.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=14.15524440000002pt/> by the formula:

<p align="center"><img src="/tex/006c3e92bb91fbf7718ece244d970192.svg?invert_in_darkmode&sanitize=true" align=middle width=235.21917704999998pt height=43.346758949999995pt/></p>

where <img src="/tex/9f7365802167fff585175c1750674d42.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=14.15524440000002pt/> and <img src="/tex/5db0aab2d6e54e70d087c4b6ae005a7a.svg?invert_in_darkmode&sanitize=true" align=middle width=15.514781699999991pt height=14.15524440000002pt/> are <img src="/tex/5824cf8e74837d9e47e5d2a445bbd200.svg?invert_in_darkmode&sanitize=true" align=middle width=29.629030199999992pt height=22.465723500000017pt/>-dimensional vectors representing clips with <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> timepoints,
<img src="/tex/02424da7f1a198554aba257b04b7f68f.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=21.95701200000001pt/> is the denoised vector/clip,

<p align="center"><img src="/tex/4bbd3c88773cd63b2ed19dc1ad35076d.svg?invert_in_darkmode&sanitize=true" align=middle width=163.4625762pt height=21.1544223pt/></p>

is a weighting function, and the summation is over all <img src="/tex/edb0013d0d98003d4407f974610d85fd.svg?invert_in_darkmode&sanitize=true" align=middle width=46.98047309999998pt height=22.465723500000017pt/> clips in the entire
recording (including all translations). Thus, each of the <img src="/tex/e908dc05eda31fa627940844fffb509c.svg?invert_in_darkmode&sanitize=true" align=middle width=41.95782194999999pt height=24.65753399999998pt/>
half-overlapping target clips <img src="/tex/9f7365802167fff585175c1750674d42.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=14.15524440000002pt/> are replaced by a weighted average of all <img src="/tex/edb0013d0d98003d4407f974610d85fd.svg?invert_in_darkmode&sanitize=true" align=middle width=46.98047309999998pt height=22.465723500000017pt/>
clips <img src="/tex/5db0aab2d6e54e70d087c4b6ae005a7a.svg?invert_in_darkmode&sanitize=true" align=middle width=15.514781699999991pt height=14.15524440000002pt/>. A back-of-the-envelope calculation reveals that the computational
burden is prohibitive for even relatively short recordings of a few minutes, but
we will outline below the various strategies we employ to overcome this
challenge in our software.

The <img src="/tex/8d590e493942863d5be1c6d34c14a9e8.svg?invert_in_darkmode&sanitize=true" align=middle width=76.19306144999999pt height=24.65753399999998pt/> term appearing in the exponent is the normalized distance between the clips given by

<p align="center"><img src="/tex/93d1b6a197ec5da5802b9f40035bfbf9.svg?invert_in_darkmode&sanitize=true" align=middle width=196.9065153pt height=17.031940199999998pt/></p>

where <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> is a <img src="/tex/b7ed1b8498c5c149928768da49ea0b89.svg?invert_in_darkmode&sanitize=true" align=middle width=64.85722319999999pt height=22.465723500000017pt/> matrix that whitens the clip vectors based on the noise model empirically derived as follows:

<p align="center"><img src="/tex/595619bb9c89bada0e1aef3314479b13.svg?invert_in_darkmode&sanitize=true" align=middle width=101.62556414999999pt height=16.66852275pt/></p>

where

<p align="center"><img src="/tex/0c3aed180c746b72ec925b5fb6942a7f.svg?invert_in_darkmode&sanitize=true" align=middle width=160.39414545pt height=18.7598829pt/></p>

is the singular-value decomposition of the <img src="/tex/5ff1317d238d952d2ddaeba5c8003f9b.svg?invert_in_darkmode&sanitize=true" align=middle width=137.79652589999998pt height=24.65753399999998pt/> matrix of all clips
and <img src="/tex/8ada1523ea4dcac0e4cf320e9e54b672.svg?invert_in_darkmode&sanitize=true" align=middle width=25.461263849999987pt height=22.465723500000017pt/> is the rank-K diagonal matrix that picks out the top <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> singular
vectors. Here, <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> is chosen to capture a user-specified percentage of the total
variance in the recording. The noise level <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/> is also empirically
estimated from the data to represent the expected distance between two noise
clips (i.e., clips without detectable spikes). The software also allows for a
scaling factor for adjusting <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/>.

As a practical matter, we need to be able to perform the above denoising
procedure within a reasonable timeframe relative to other processing steps.
Below we discuss several strategies we use to speed up the computation.

### Denoising in time blocks

The first simplification is to split the recording into discrete time blocks,
typically 30 seconds or a minute, and denoise independently in each block.
For firing rates greater around 1 Hz or higher this is okay since we only need
a few dozen representative events for each neuron. In the future we may provide
a way to also probe beyond the boundaries of the discrete blocks, but for now
the user must choose a fixed duration for block sizes, with a tradeoff between
block duration and computational efficiency.

### Adaptive subsampling

The time-consuming part of the non-local means formula is the summation over all
<img src="/tex/0b7a5a5269a5ccb317f84ddae5058d9c.svg?invert_in_darkmode&sanitize=true" align=middle width=80.87312969999998pt height=22.465723500000017pt/> clips in the time block of size <img src="/tex/7acf3dadc6a35fc888add78e53dc6861.svg?invert_in_darkmode&sanitize=true" align=middle width=19.760314199999993pt height=22.465723500000017pt/>. Fortunately, the weighted
average can be viewed as a statistical procedure which may be substantially sped
up using strategic subsampling. Because firing events are usually sparse, the
majority of clips <img src="/tex/9f7365802167fff585175c1750674d42.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=14.15524440000002pt/> are close to the backround noise and therefore have a
very large number of nearby neighbors <img src="/tex/5db0aab2d6e54e70d087c4b6ae005a7a.svg?invert_in_darkmode&sanitize=true" align=middle width=15.514781699999991pt height=14.15524440000002pt/> such that <img src="/tex/c8879850e26439fbbb536d113dda8587.svg?invert_in_darkmode&sanitize=true" align=middle width=63.533307749999985pt height=24.65753399999998pt/> is close to
the maximum of <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>. For these cases it is acceptable to perform vast
subsampling, perhaps summing over only a couple hundred randomly-selected clips.
On the other extreme, some clips <img src="/tex/9f7365802167fff585175c1750674d42.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=14.15524440000002pt/> may include spikes that are relatively
rare, and thus there will be a small number of source clips that contribute anything
substantial to the sum.

While algorithms such as clustering or k-nearest neighbors could be used to more intelligently sample, we would like to avoid these types of methods in this
preprocessing step. We view clustering and classification as part of the spike
sorting step, and not this denoising operation which seeks only to isolate the
signal from the noise.

The procedure we use for adaptive subsampling involves computing the summation
in batches and selectively dropping out clips from both the sources (<img src="/tex/5db0aab2d6e54e70d087c4b6ae005a7a.svg?invert_in_darkmode&sanitize=true" align=middle width=15.514781699999991pt height=14.15524440000002pt/>) and
the targets (<img src="/tex/9f7365802167fff585175c1750674d42.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=14.15524440000002pt/>) between each batch. In the first batch we compute
<img src="/tex/3e6cdda6bcbb97258f8f836fe15c3f46.svg?invert_in_darkmode&sanitize=true" align=middle width=76.19306144999999pt height=24.65753399999998pt/> for all <img src="/tex/9f7365802167fff585175c1750674d42.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61896569999999pt height=14.15524440000002pt/>'s and a small subset of the <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>'s and seperately
accumulate both the numerator <img src="/tex/70c1c0684cd2c896667c058061245354.svg?invert_in_darkmode&sanitize=true" align=middle width=104.61219779999999pt height=24.657735299999988pt/> and the denominator
<img src="/tex/c507c62694721a6c00a2a9126404d569.svg?invert_in_darkmode&sanitize=true" align=middle width=89.09741609999999pt height=24.657735299999988pt/>. We then use the size of the denominator as a criterion for
determining which clips to exclude from the subsequent batches, the idea being
that a large denominator means that a large number of nearby neighbors have
already contributed to the weighted average. The user sets a threshold (e.g.,
30) for dropping out clips in this way.

In addition to dropping out target clips from the computation, it is crucial to
also drop out source clips. A large denominator for a target clip means that it
must have a relatively large number of nearby neighbors. Therefore, the source
clips that are overlapping (in time) to such a target clip would presumably also
have a large number of neighbors, and in fact all of its nearby neighbors would
be expected to have a large number of nearby neighbors. Thus it should be safe
to drop out source clips as well based on the denominator criterion for the
time-overlapping target clips.

In summary, adaptive subsampling is achieved by computing the weighted sum by
accumulating the numerators and denominators in batches, while dropping out both
source and target clips based on the denominator threshold criterion.

### Combining overlapping clips

Here we need to describe how we keep track of numerators and denominators for
each target clip, how exactly we apply the denominator dropout criterion, and
how to combine the values for time-overlapping clips.

### Overlapping spikes and other rare events

The method of non-local means works well when the signal repeats many times
throughout the time block. But when neural events overlap in time the resulting
waveform is a superposition that will usually not match any other waveform.
Even if the same two neurons fire simultaneously in multiple instances, the
time offset between the two events is expected to be variable, thus producing
a spectrum of different superpositioned waveforms. Overlapping spike events are
thus expected to form isolated clips that have few if any nearby neighbors.

While our method cannot be expected to denoise such events, we can expect the
noisy signal of the superimposed waveforms to survive the denoising process.
This is because only one source term (the clip itself) is expected to contribute
substantially (always with a weight of <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>) to the average. Therefore, we
expect that the denoised recording will retain overlapping or other rare events,
but without denoising them.

### Large channel arrays

Describe the procedure for denoising in neighborhoods.

### GPU processing

Describe how we use PyTorch to efficiently compute the matrix-matrix
multiplications needed in the above-described algorithm.

## Prerequisites

* Python (tested on 3.6 and 3.7)
* [PyTorch](https://pytorch.org/) (tested on v1.0.0)
* [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) - if using GPU (recommended) 
* [MKL](https://software.intel.com/en-us/mkl) - if using CPU instead of CUDA

To test whether PyTorch and CUDA are set up properly, run the following in ipython:

```python
import torch
if torch.cuda.is_available():
    print('CUDA is available for PyTorch!')
else:
    print('CUDA is NOT available for PyTorch.')
```

**Recommended**

* [SpikeInterface](https://github.com/spikeinterface) -- `pip install spikeinterface`
* [SpikeForest](https://github.com/flatironinstitute/spikeforest2)

## Install from PyPI

```
pip install --upgrade ephys_nlm
```

## Install from source (for developers)

After cloning this repository:

```bash
cd ephys_nlm
pip install -e .

# Then in subsequent updates:
git pull
pip install -e .
```

# Example

The following is taken from a notebook in the [examples/](examples/) directory. It generates a short synthetic recording and denoise it.

```python
from ephys_nlm import ephys_nlm_v1, ephys_nlm_v1_opts

import spikeextractors as se
import spikewidgets as sw
import matplotlib.pyplot as plt

# Create a synthetic recording for purposes of demo
recording, sorting_true = se.example_datasets.toy_example(duration=30, num_channels=4, K=20, seed=4)

# Specify the denoising options
opts = ephys_nlm_v1_opts(
    multi_neighborhood=False, # False means all channels will be denoised in one neighborhood
    block_size_sec=30, # Size of block in seconds -- each block is denoised separately
    clip_size=30, # Size of a clip (aka snippet) in timepoints
    sigma='auto', # Auto compute the noise level
    sigma_scale_factor=1, # Scale factor for auto-computed noise level
    whitening='auto', # Auto compute the whitening matrix
    whitening_pctvar=90, # Percent of variance to retain - controls number of SVD components to keep
    denom_threshold=30 # Higher values lead to a slower but more accurate calculation.
)

# Do the denoising
recording_denoised, runtim_info = ephys_nlm_v1(
    recording=recording,
    opts=opts,
    device='cpu', # cuda is recommended for non-demo situations
    verbose=1
)
```

Also included in the notebook is SpikeInterface code used to view the original and denoised timeseries:

```python
# View the original and denoised timeseries

plt.figure(figsize=(16,5))
sw.TimeseriesWidget(recording=recording, trange=(0, 0.2), ax=plt.gca()).plot();

plt.figure(figsize=(16,5))
sw.TimeseriesWidget(recording=recording_denoised, trange=(0, 0.2), ax=plt.gca()).plot();
```

This should produce output similar to the following:

![screenshot1](doc/screenshot1.png)

## License

Apache-2.0 -- We request that you acknowledge the original authors in any derivative work.

## Authors

Jeremy Magland, Center for Computational Mathematics (CCM), Flatiron Institute

## Acknowledgments

Alex Barnett, James Jun, and members of CCM for many useful discussions