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

is the singular-value decomposition of the <img src="/tex/67b4fdd3319785612314e934a155112e.svg?invert_in_darkmode&sanitize=true" align=middle width=109.48612455pt height=24.65753399999998pt/> matrix of all clips
and <img src="/tex/8ada1523ea4dcac0e4cf320e9e54b672.svg?invert_in_darkmode&sanitize=true" align=middle width=25.461263849999987pt height=22.465723500000017pt/> is the rank-K diagonal matrix that picks out the top <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> singular
vectors. Here, <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> is chosen to capture a user-specified percentage of the total
variance in the recording. The noise level <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/> is also empirically
estimated from the data to represent the expected distance between two noise
clips (i.e., clips without detectable spikes). The software also allows for a
scaling factor for adjusting <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/>.

As a practical matter, we need to be able to perform the above denoising
procedure within a reasonable timeframe relative to other processing steps.
We use the following strategies to speed up the computation.

### Denoising in time blocks

The first simplification is to split the recording into discrete time blocks,
typically 30 seconds or a minute, and denoise independently in each block.
For firing rates greater around 1 Hz or higher this is okay since we only need
a few dozen representative events for each neuron. In the future we may provide
a way to also probe beyond the boundaries of the discrete blocks, but for now
the user must choose a fixed duration for block sizes, with a tradeoff between
block duration and computational efficiency.

### Adaptive subsampling

TODO

### GPU processing

TODO






TODO: finish writing this section

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

## Install from source

For now, install in development mode. After cloning this repository:

```bash
cd ephys_nlm
pip install -e .
```

Then in subsequent updates:

```bash
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

## Authors

Jeremy Magland, Center for Computational Mathematics, Flatiron Institute