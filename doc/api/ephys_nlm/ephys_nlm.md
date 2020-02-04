Module ephys_nlm.ephys_nlm
==========================
Core routines for ephys_nlm

Authors: Jeremy Magland, Center for Computational Mathematics, Flatiron Institute

Created January 2019

Functions
---------

    
`ephys_nlm_v1(recording, *, opts, device, verbose=1)`
:   Denoise an ephys recording using non-local means.
    
    The input and output recordings are RecordingExtractors from SpikeInterface.
    
    Parameters
    ----------
    recording : se.RecordingExtractor
        The ephys recording to denoise (see SpikeInterface)
    opts : EphysNlmV1Opts
        Options created using EphysNlmV1Opts(...)
    device : Union[str, None]
        Either cuda or cpu (cuda is highly recommended, but you need to have
        CUDA/PyTorch working on your system). If None, then the EPHYS_NLM_DEVICE
        environment variable will be used.
    verbose : int, optional
        Verbosity level, by default 1
    
    Returns
    -------
    Tuple[OutputRecordingExtractor, EphysNlmV1Info]
        The output recording extractor see SpikeInterface
        and info about the run

    
`ephys_nlm_v1_opts(multi_neighborhood=False, neighborhood_adjacency_radius=None, block_size=None, block_size_sec=30, clip_size=30, sigma='auto', sigma_scale_factor=1, whitening='auto', whitening_pctvar=90, denom_threshold=30)`
:   Create options to be passed into ephys_nlm_v1.
    
    These options affect the calculation performed, not the environment. So for
    example, device is not among these options.
    
    Parameters
    ----------
    multi_neighborhood : bool, optional
        Whether to denoise multiple neighborhoods. If False (the default), all channels will be denoised in one neighorhood.
    neighborhood_adjacency_radius : int, optional
        Adjacency radius used to determine neighborhoods if multi_neighborhood is True, by default None
    block_size : Union[int, None], optional
        Size of blocks (in timepoints). Each block will be denoised separately. If None (the default), block_size_sec will be used to compute block_size.
    block_size_sec : Union[float, None], optional
        If block_size is None, use this to specify the block size in seconds. Each block will be denoised separately. The default is 30 seconds.
    clip_size : int, optional
        Size of a clip (aka snippet) in timepoints. The default is 30.
    sigma : str, optional
        Method for computing the noise level. Right now the only option is 'auto', which is also the default.
    sigma_scale_factor : float, optional
        A scale factor to adjust the auto-computed noise level, by default 1
    whitening : str, optional
        Method for computing the whitening matrices. Right now the only option is 'auto', which is also the default.
    whitening_pctvar : float, optional
        The percent of variance to retain in the whitening matrix controlling the number of SVD components to keep. The default is 90.
    denom_threshold : float, optional
        A threshold controlling the maximum number of clips to use to denoise each clip. The default is 30. Higher values lead to a slower but more accurate calculation.

Classes
-------

`EphyNlmV1BlockInfo()`
:   Run info for a block

`EphysNlmV1Info()`
:   Run info returned by ephys_nlm_v1

`EphysNlmV1NeighborhoodInfo()`
:   Run info for a neighborhood

`EphysNlmV1Opts()`
:   Denoising options to be passed into ephys_nlm_v1. These options affect the calculation performed, not the environment. So for example, device is not among these options.