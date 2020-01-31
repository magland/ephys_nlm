# ephys_nlm

**----------- Under development -----------**

Non-local means denoising of multi-channel electrophysiology timeseries using PyTorch.

## Prerequisites

* Python (tested on v3.6)
* PyTorch (tested on v1.0.0)
* CUDA - if using GPU (recommended) 
* MKL - if not using GPU

**Recommended**

* spikeinterface -- `pip install spikeinterface`
* spikeforest2

## Install from source

For now, install in development mode. After cloning this repository:

```
cd ephys_nlm
pip install -e .
```

Then in subsequent updates:

```
git pull
pip install -e .
```

# Example

```python
import spikeextractors as se
from ephys_nlm import ephys_nlm, ephys_nlm_opts

# Create a synthetic recording for purposes of demo
recording, sorting_true = se.example_datasets.toy_example()

opts = ephys_nlm_opts(
    multi_neighborhood=False, # All channels in one neighborhood
    block_size=recording.get_sampling_frequency() * 30, # Size of denoising blocks (num. timepoints)
    clip_size=30, # Size of clip size for denoising (num. timepoints)
    sigma='auto', # Auto determine noise level
    whitening_matrix='auto', # Auto compute whitening matrix based on data
    denom_threshold=30 # Higher values => slower but more accurate
)
recording_denoised = ephys_nlm(recording=recording, opts=opts)

```
