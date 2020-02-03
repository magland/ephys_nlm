[![Build Status](https://travis-ci.org/magland/ephys_nlm.svg?branch=master)](https://travis-ci.org/magland/ephys_nlm)

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

is the singular-value decomposition of the $(N-T+1) \times MT$ matrix of all clips
and $D_K$ is the rank-K diagonal matrix that picks out the top $K$ singular
vectors. Here, $K$ is chosen to capture a user-specified percentage of the total
variance in the recording. The noise level $\sigma$ is also empirically
estimated from the data to represent the expected distance between two noise
clips (i.e., clips without detectable spikes). The software also allows for a
scaling factor for adjusting $\sigma$.

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
$N_0-T+1$ clips in the time block of size $N_0$. Fortunately, the weighted
average can be viewed as a statistical procedure which may be substantially sped
up using strategic subsampling. Because firing events are usually sparse, the
majority of clips $v_i$ are close to the backround noise and therefore have a
very large number of nearby neighbors $u_j$ such that $w(u_j, v_j)$ is close to
the maximum of $1$. For these cases it is acceptable to perform vast
subsampling, perhaps summing over only a couple hundred randomly-selected clips.
On the other extreme, some clips $v_i$ may include spikes that are relatively
rare, and thus there will be a small number of source clips that contribute anything
substantial to the sum.

While algorithms such as clustering or k-nearest neighbors could be used to more intelligently sample, we would like to avoid these types of methods in this
preprocessing step. We view clustering and classification as part of the spike
sorting step, and not this denoising operation which seeks only to isolate the
signal from the noise.

The procedure we use for adaptive subsampling involves computing the summation
in batches and selectively dropping out clips from both the sources ($u_j$) and
the targets ($v_i$) between each batch. In the first batch we compute
$\|v_i-u_j\|_A$ for all $v_i$'s and a small subset of the $j$'s and seperately
accumulate both the numerator $\sum_j w(v_i, u_j)u_j$ and the denominator
$\sum_j w(v_i, u_j)$. We then use the size of the denominator as a criterion for
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
substantially (always with a weight of $1$) to the average. Therefore, we
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