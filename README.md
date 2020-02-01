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
images, but it can also be adapted for use with 1d signals. Let $y_{m, t}$ be
the multi-channel voltage trace for $m = 1,...,M$ and $t=1,...,N$ where $M$ is
the number of electrode channels, and $N$ is the number of timepoints. If
computation time were not a concern, we would ideally denoise each half-overlapping clip (or snippet) $v_i$ by the formula:

$$\tilde{v}_i = \frac{1}{\sum_j w(v_i, u_j)}\sum_j w(v_i, u_j) u_j$$

where $v_i$ and $u_j$ are $MT$-dimensional vectors representing clips with $T$ timepoints,
$\tilde{v}_i$ is the denoised vector/clip,

$$w(i, j) = e^{-\|v_i-v_j\|_A^2/\sigma^2}$$

is a weighting function, and the summation is over all $N-T$ clips in the entire
recording (including all translations). Thus, each of the $2N/T$
half-overlapping target clips $v_i$ are replaced by a weighted average of all $N-T$
clips $u_j$. A back-of-the-envelope calculation reveals that the computational
burden is prohibitive for even relatively short recordings of a few minutes, but
we will outline below the various strategies we employ to overcome this
challenge in our software.

The $\|v_i - u_j\|_A$ term appearing in the exponent is the normalized distance between the clips given by

$$\|v_i - u_j\|_A = \|(v_i - u_j)A\|_2$$

where $A$ is a $MT \times K$ matrix that whitens the clip vectors based on the noise model empirically derived as follows:

$$A = V S^{-1} D_K$$

where

$$USV^T = [u_1 ... u_{N-T}]^T$$

is the singular-value decomposition of the $(N-T) \times MT$ matrix of all clips
and $D_K$ is the rank-K diagonal matrix that picks out the top $K$ singular
vectors. Here, $K$ is chosen to capture a user-specified percentage of the total
variance in the recording. The noise level $\sigma$ is also empirically
estimated from the data to represent the expected distance between two noise
clips (i.e., clips without detectable spikes). The software also allows for a
scaling factor for adjusting $\sigma$.

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