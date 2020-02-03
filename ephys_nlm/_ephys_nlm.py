"""Core routines for ephys_nlm

Authors: Jeremy Magland, Center for Computational Mathematics, Flatiron Institute

Created January 2019
"""

import os
import time
import math
from typing import Tuple, List, Dict, Union, Any
import numpy as np
import torch
import spikeextractors as se
from ._outputrecordingextractor import OutputRecordingExtractor


def ephys_nlm_v1_opts(
        multi_neighborhood: bool = False,
        neighborhood_adjacency_radius: Union[float, None] = None,
        block_size: Union[int, None] = None,
        block_size_sec: Union[float, None] = 30,
        clip_size: int = 30,
        sigma: str = 'auto',
        sigma_scale_factor: float = 1,
        whitening: str = 'auto',
        whitening_pctvar: float = 90,
        denom_threshold: float = 30
):
    """Create options to be passed into ephys_nlm_v1. These options affect the calculation performed, not the environment. So for example, device is not among these options.

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
    """

    opts = EphysNlmV1Opts()

    opts.multi_neighborhood = multi_neighborhood
    opts.neighborhood_adjacency_radius = neighborhood_adjacency_radius
    opts.block_size = block_size
    opts.block_size_sec = block_size_sec
    opts.clip_size = clip_size
    opts.sigma = sigma
    opts.sigma_scale_factor = sigma_scale_factor
    opts.whitening = whitening
    opts.whitening_pctvar = whitening_pctvar
    opts.denom_threshold = denom_threshold

    return opts


class EphysNlmV1Opts:
    """Denoising options to be passed into ephys_nlm_v1. These options affect the calculation performed, not the environment. So for example, device is not among these options.
    """

    def __init__(self):
        # Note: These are not the defaults -- the defaults are defined in ephys_nlm_v1_opts
        self.multi_neighborhood: bool = False
        self.neighborhood_adjacency_radius: Union[float, None] = None
        self.block_size: Union[int, None] = None
        self.block_size_sec: Union[float, None] = None
        self.clip_size: int = 0
        self.sigma: str = ''
        self.sigma_scale_factor: float = 0
        self.whitening: str = ''
        self.whitening_pctvar: float = 0
        self.denom_threshold: float = 0

        # Internal for convenience
        self._device: str = ''


class EphysNlmV1Info:
    """Run info returned by ephys_nlm_v1
    """
    def __init__(self):
        self.opts: Union[EphysNlmV1Opts, None] = None
        self.recording: Any = None
        self.device: str = ''
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed_time: float = 0
        self.blocks: List[EphyNlmV1BlockInfo] = []


class EphyNlmV1BlockInfo:
    """Run info for a block
    """
    def __init__(self):
        self.neighborhoods: List[EphysNlmV1NeighborhoodInfo] = []


class EphysNlmV1NeighborhoodInfo:
    """Run info for a neighborhood
    """
    def __init__(self):
        pass


def ephys_nlm_v1(recording: se.RecordingExtractor, *, opts: EphysNlmV1Opts, device: Union[None, str], verbose: int = 1) -> Tuple[OutputRecordingExtractor, EphysNlmV1Info]:
    """Denoise an ephys recording using non-local means

    Parameters
    ----------
    recording : se.RecordingExtractor
        The ephys recording to denoise (see SpikeInterface)
    opts : EphysNlmV1Opts
        Options created using EphysNlmV1Opts(...)
    device : str
        Either cuda or cpu (cuda is highly recommended, but you need to have CUDA/PyTorch working on your system)
    verbose : int, optional
        Verbosity level, by default 1

    Returns
    -------
    Tuple[OutputRecordingExtractor, EphysNlmV1Info]
        The output recording extractor see SpikeInterface
        and info about the run
    """
    channel_ids = recording.get_channel_ids()
    M = len(channel_ids)
    N = recording.get_num_frames()
    T = opts.clip_size
    assert T % 2 == 0, 'clip size must be divisible by 2.'
    info = EphysNlmV1Info()
    info.recording = recording
    info.opts = opts
    info.device = device
    info.start_time = time.time()
    if opts.block_size is None:
        if opts.block_size_sec is None:
            raise Exception('block_size and block_size_sec are both None')
        opts.block_size = int(
            recording.get_sampling_frequency() * opts.block_size_sec)
    block_size = opts.block_size
    N = recording.get_num_frames()
    num_blocks = max(1, math.floor(N / block_size))
    assert opts.sigma == 'auto', 'Only sigma=auto allowed at this time'
    assert opts.whitening == 'auto', 'Only whitening=auto allowed at this time'

    if device is None:
        device = os.getenv('EPHYS_NLM_DEVICE', None)
        if device is None or device0 == '':
            print('Warning: EPHYS_NLM_DEVICE not set -- defaulting to cpu. To use GPU, set EPHYS_NLM_DEVICE=cuda')
            device = 'cpu'
    elif device == 'cpu':
        print('Using device=cpu. Warning: GPU is much faster. To use GPU, set device=cuda')
    if device == 'cuda':
        assert torch.cuda.is_available(), f'Cannot use device=cuda. PyTorch/CUDA is not configured properly -- torch.cuda.is_available() is returning False.'
        print('Using device=cuda')
    elif device == 'cpu':
        print('Using device=cpu')
    else:
        raise Exception(f'Invalid device: {device}')
    opts._device = device  # for convenience

    neighborhoods = _get_neighborhoods(recording=recording, opts=opts)

    if verbose >= 1:
        print(f'Denoising recording of size {M} x {N} using {len(neighborhoods)} neighborhoods and {num_blocks} time blocks')

    initial_traces = recording.get_traces(
        start_frame=0, end_frame=min(N, block_size))
    _estimate_sigma_and_whitening(
        traces=initial_traces, neighborhoods=neighborhoods, opts=opts, verbose=verbose)

    recording_out = OutputRecordingExtractor(
        base_recording=recording, block_size=opts.block_size)

    for ii in range(num_blocks):
        if verbose >= 1:
            print(f'Denoising block {ii} of {num_blocks}')

        t1 = ii * block_size
        t2 = t1 + block_size
        # The last block is potentially larger
        if ii == num_blocks - 1:
            t2 = N

        block_traces = recording.get_traces(start_frame=t1, end_frame=t2)
        block_traces_denoised, block_info = _denoise_block(
            traces=block_traces,
            opts=opts,
            neighborhoods=neighborhoods,
            verbose=verbose
        )
        info.blocks.append(block_info)
        recording_out.add_block(block_traces_denoised)

    info.end_time = time.time()
    info.elapsed_time = info.end_time - info.start_time

    return recording_out, info


def _estimate_sigma_and_whitening(*, traces: np.ndarray, neighborhoods: List[Dict], opts: EphysNlmV1Opts, verbose: int):
    """Estimate sigma and the whitening matrix in all neighborhoods based on a chunk of data. It will add the information to the neighborhoods.

    Parameters
    ----------
    traces : np.ndarray
        The M x N array of traces
    neighborhoods : List[Dict]
        A list of neighborhoods -- the output gets added here
    opts : EphysNlmV1Opts
        Denoising options
    verbose : int
        Verbosity level
    """
    for neighborhood in neighborhoods:
        traces_n = traces[neighborhood['channel_indices'], :]
        sigma, whitening_matrix = _estimate_sigma_and_whitening_in_neighborhood(
            traces=traces_n, opts=opts, verbose=verbose)
        neighborhood['sigma'] = sigma * opts.sigma_scale_factor
        neighborhood['whitening_matrix'] = whitening_matrix


def _estimate_sigma_and_whitening_in_neighborhood(*, traces: np.ndarray, opts: EphysNlmV1Opts, verbose: int) -> Tuple[float, np.ndarray]:
    """Estimate sigma and whitening matrix in a neighborhood based on a chunk of data.

    Parameters
    ----------
    traces : np.ndarray
        The M x N array of traces in a neighborhood
    opts : EphysNlmV1Opts
        Denoising options
    verbose : int
        Verbosity level

    Returns
    -------
    Tuple[float, np.ndarray]
        Sigma (noise level) and the whitening matrix (MT x K), where K is the number of components to retain
    """
    M = traces.shape[0]
    N = traces.shape[1]
    T = opts.clip_size
    device = opts._device
    num_clips_for_estimating_whitening = 1000
    num_clips = min(num_clips_for_estimating_whitening, N-T)
    L = num_clips
    assert T %2 == 0
    T_delta = int(T / 2)
    whitening_pctvar = opts.whitening_pctvar

    ###############################################################################################################
    # First we estimate the whitening matrix
    if verbose >= 3:
        print('Estimating whitening matrix...')

    # M x N
    traces_t = torch.from_numpy(traces).to(device)
    # L x M x T
    clips_t = extract_clips_t(
        traces_t, t_start=0, T_delta=T_delta, T=T, num_clips=L)
    del traces_t
    # L x MT
    X_t = clips_t.reshape((L, M*T))
    # normsqrs = torch.sum(X_t**2, dim=[1])
    # _U: L x MT
    # S: MT
    # V: MT x MT
    _U, S, V = torch.svd(X_t)

    # MT
    S_np = S.data.cpu().numpy()
    fracvar0 = np.cumsum(S_np)/np.sum(S_np)
    ncomp = np.where(fracvar0 >= whitening_pctvar/100)[0][0] + 1
    print(
        f'Using {ncomp} components to explain {fracvar0[ncomp-1]*100} percent of variance in neighborhood')

    # MT x ncomp
    whitening_matrix = torch.mm(
        V[:, :ncomp], torch.diag_embed(S[:ncomp]**(-1)))
    ###############################################################################################################

    ###############################################################################################################
    # Now that we have the whitening matrix, let's estimate sigma
    if verbose >= 3:
        print('Estimating sigma...')

    Xw_t = torch.mm(X_t, whitening_matrix)
    # L
    normsqrXw = torch.sum(Xw_t**2, dim=[1])
    # L x L
    inner_products = torch.mm(Xw_t, Xw_t.t())
    # L x L
    distsqrs = normsqrXw.reshape((L, 1)).repeat(
        (1, L)) + normsqrXw.reshape((1, L)).repeat((L, 1)) - 2*inner_products
    distsqrs = distsqrs.data.cpu().numpy()

    vals = np.median(distsqrs, axis=1)
    cutoff = np.median(vals)
    inds = np.nonzero(vals <= cutoff)[0]
    vals = distsqrs[inds][:, inds]
    for i in range(len(inds)):
        vals[i, i] = 0
    vals = vals.reshape(len(inds)**2)
    vals = vals[np.where(vals > 0)[0]]
    vals = np.sqrt(vals)
    sigma = np.median(vals)
    plot_hist = False
    if plot_hist:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(vals, 100)
    ###############################################################################################################

    return sigma, whitening_matrix


def _denoise_block(*, traces: np.ndarray, opts: EphysNlmV1Opts, neighborhoods: List[Dict], verbose: int) -> Tuple[np.ndarray, EphyNlmV1BlockInfo]:
    """Denoise all neighborhoods in a block of data in the recording (e.g., one minute)

    Parameters
    ----------
    traces : np.ndarray
        The M x N array of ephys traces
    opts : EphysNlmV1Opts
        Denoising options
    neighborhoods : List[Dict]
        The list of neighborhoods, each containing information about channel indices, sigma, and whitening matrix
    verbose : int
        Verbosity level

    Returns
    -------
    Tuple[np.ndarray, EphyNlmV1BlockInfo]
        The denoised chunk of data (M x N) and info about the run
    """
    M = traces.shape[0]
    N = traces.shape[1]

    info = EphyNlmV1BlockInfo()
    traces_denoised = np.zeros((M, N), dtype=np.float32)

    for neighborhood in neighborhoods:
        channel_indices = neighborhood['channel_indices']
        target_indices = neighborhood['target_indices']
        sigma = neighborhood['sigma']
        whitening_matrix = neighborhood['whitening_matrix']
        traces_denoised0, info0 = _denoise_neighborhood(
            traces=traces[channel_indices, :], sigma=sigma, whitening_matrix=whitening_matrix, opts=opts, verbose=verbose)
        for ii in target_indices:
            ii2 = np.nonzero(channel_indices == ii)[0][0]
            traces_denoised[ii, :] = traces_denoised0[ii2, :]
        info.neighborhoods.append(info0)

    return traces_denoised, info


def _denoise_neighborhood(*, traces: np.ndarray, opts: EphysNlmV1Opts, sigma: float, whitening_matrix: np.ndarray, verbose: int) -> Tuple[np.ndarray, EphysNlmV1NeighborhoodInfo]:
    """Denoise a chunk of data in a neighborhood of the ephys recording

    Parameters
    ----------
    traces : np.ndarray
        Input recording data (M x N)
    opts : EphysNlmV1Opts
        Denoising options
    sigma : float
        Noise level
    whitening_matrix : np.ndarray
        Whitening matrix (MT x K), where K is the number of components to retain
    verbose : int
        Verbosity level

    Returns
    -------
    Tuple[np.ndarray, EphysNlmV1NeighborhoodInfo]
        The denoised traces (M x N) and info about the run
    """
    M = traces.shape[0]
    N = traces.shape[1]
    T = opts.clip_size
    device = opts._device
    assert T % 2 == 0, 'clip size must be divisible by 2.'
    T2 = int(T/2)
    num_clips = math.ceil(N / T2)
    N2 = T2 * num_clips + T
    L = num_clips
    assert whitening_matrix.shape[0] == M * \
        T, 'Unexpected first dimension for whitening matrix in neighborhood'
    device = opts._device
    denom_threshold = opts.denom_threshold

    if verbose >= 2:
        print(f'Denoising neighborhood ({M} x {N})')

    info = EphysNlmV1NeighborhoodInfo()

    traces_t = torch.zeros((M, N2), dtype=torch.float32, device=device)
    traces_t[:, :N] = torch.from_numpy(traces).to(device)
    # L x M x T
    clips_t = extract_clips_t(
        traces_t, t_start=0, T_delta=T2, T=T, num_clips=L)

    numer_t = torch.zeros((L, M, T), dtype=torch.float32, device=device)
    denom_t = torch.zeros((L), dtype=torch.float32, device=device)

    for dt in range(T2):
        if dt == 0:
            # for the first step (dt=0) we use a subsampling factor
            subsample_factor = 10
        else:
            subsample_factor = 1
        clips2_t = extract_clips_t(
            traces_t, t_start=dt, T_delta=T2, T=T, num_clips=L)
        for aa in range(subsample_factor):
            inds_subsample_t = torch.arange(aa, L, subsample_factor)
            denom_scores_t = torch.min(
                denom_t[:L-2]+denom_t[1:L-1], denom_t[1:L-1]+denom_t[2:L])
            denom_scores_t = torch.cat(
                (denom_t[:1], denom_scores_t, denom_t[L-1:]))
            selected_inds_t = torch.nonzero(
                denom_scores_t <= denom_threshold).squeeze()
            # consider: take into consideration dt so we would have different selected inds for clips2
            selected_inds2_t = torch.nonzero(
                denom_scores_t[inds_subsample_t] <= denom_threshold).squeeze()
            selected_inds2_t = inds_subsample_t[selected_inds2_t]
            Lsel = len(selected_inds_t)
            Lsel2 = len(selected_inds2_t)
            if Lsel == 0:
                break
            if Lsel > 0 and Lsel2 > 0:
                if verbose >= 3:
                    print(
                        f'dt={dt}; aa={aa}; Num selected clips: {Lsel}/{Lsel2}')
                clips_sel_t = clips_t[selected_inds_t, :, :]
                clips2_sel_t = clips2_t[selected_inds2_t, :, :]

                numer0_t, denom0_t = _denoise_clips_t(
                    clips_sel_t, clips2_sel_t,
                    sigma=sigma,
                    whitening_matrix=whitening_matrix,
                    verbose=verbose,
                    maxnum=None,
                    device=device
                )
                numer_t[selected_inds_t, :, :] += numer0_t
                denom_t[selected_inds_t] += denom0_t
        subsample_factor = 1

    numer_cpu = numer_t.data.cpu().numpy()
    denom_cpu = denom_t.data.cpu().numpy()
    numer_timeseries = np.zeros((M, N+T), dtype=np.float32)
    denom_timeseries = np.zeros((N+T), dtype=np.float32)
    for ii in range(L):
        numer_timeseries[:, ii*T2:ii*T2+T] += numer_cpu[ii, :, :]
        denom_timeseries[ii*T2:ii*T2+T] += denom_cpu[ii]
    traces_denoised = numer_timeseries[:, :N]/(0.000001 + denom_timeseries[:N])

    return traces_denoised, info


def _denoise_clips_t(X_t: torch.tensor, Y_t: torch.tensor, *, sigma: float, whitening_matrix: torch.tensor, verbose: int, maxnum: Union[None, int], device: str) -> Tuple[torch.tensor, torch.tensor]:
    """Denoise clips based on other clips, assembling a numerator and a denominator for use in the calling function. The function calls denoise_clips_t_B().

    Parameters
    ----------
    X_t : torch.tensor
        Clips to denoise (L1 x M x T)
    Y_t : torch.tensor
        Clips to use for denoising (L2 x M x T)
    sigma : float
        Noise level
    whitening_matrix : torch.tensor
        Whitening matrix (MT x K)
    verbose : int
        Verbosity level
    maxnum : Union[None, int]
        Optional maximum number of clips to use to denoise each clip. The default, None, means use as many neighbors as are available
    device : str
        The device: cuda or cpu

    Returns
    -------
    Tuple[torch.tensor, torch.tensor]
        The numerator (L1 x M x T) and denominator (L1)
    """
    # X_t: L1 x M x T
    # Y_t: L2 x M x T
    # output:
    #   numer_t: L1 x M x T
    #   denom_t: L1
    L1 = X_t.shape[0]
    L2 = Y_t.shape[0]
    M = X_t.shape[1]
    T = Y_t.shape[2]
    assert X_t.shape[2] == T
    assert Y_t.shape[1] == M
    assert whitening_matrix.shape[0] == M*T

    # L1 x K
    Xw_t = torch.mm(X_t.reshape((L1, M * T)), whitening_matrix)
    # L2 x K
    Yw_t = torch.mm(Y_t.reshape((L2, M * T)), whitening_matrix)

    # TODO: make this a parameter -- this affects memory usage
    max_size = 2000000  # affects memory usage
    numer_t = torch.zeros((L1, M, T), dtype=torch.float32, device=device)
    denom_t = torch.zeros((L1), dtype=torch.float32, device=device)
    num_subproblems = math.ceil(L1*L2/(max_size+1))
    # print(f'Splitting into {num_subproblems} subproblems')
    increment = int(L2 / num_subproblems)
    for ii in range(0, L2, increment):
        ii2 = min(ii+increment, L2)
        numer0, denom0 = _denoise_clips_t_B(
            Xw_t=Xw_t,
            Yw_t=Yw_t[ii:ii2, :],
            Y_t=Y_t[ii:ii2, :, :],
            sigma=sigma,
            verbose=verbose,
            maxnum=maxnum,
            device=device
        )
        numer_t += numer0
        denom_t += denom0
    return numer_t, denom_t


def _denoise_clips_t_B(Xw_t: torch.tensor, Yw_t: torch.tensor, Y_t: torch.tensor, *, sigma: float, verbose: int, maxnum: Union[None, int], device: str) -> Tuple[torch.tensor, torch.tensor]:
    """The work engine of denoise_clips_t taking in the transformed/whitened data

    Parameters
    ----------
    Xw_t : torch.tensor
        Transformed/whitened clips to denoise (L1 x K)
    Yw_t : torch.tensor
        Transformed/whitened clips to use for denoising (L2 x K)
    Y_t : torch.tensor
        Original clips to use for denoising (L2 x M x T)
    sigma : float
        Noise level
    verbose : int
        Verbosity level
    maxnum : Union[None, int]
        Optional maximum number of clips to use to denoise each clip. The default, None, means use as many neighbors as are available
    device : str
        The device: cuda or cpu

    Returns
    -------
    Tuple[torch.tensor, torch.tensor]
        The numerator (L1 x M x T) and denominator (L1)
    """

    L1 = Xw_t.shape[0]
    K = Xw_t.shape[1]
    L2 = Yw_t.shape[0]
    M = Y_t.shape[1]
    T = Y_t.shape[2]
    assert Y_t.shape[0] == L2
    assert Yw_t.shape[1] == K

    if verbose >= 5:
        print(f'Denoising clip {L1} clips using {L2} clips (M={M}, T={T}, K={K})')

    # L1 x L2
    inner_products = torch.mm(Xw_t, Yw_t.t())

    # L1
    normsqrX = torch.sum(Xw_t**2, dim=[1])

    # L2
    normsqrY = torch.sum(Yw_t**2, dim=[1])

    # L1 x L2
    distsqr = normsqrX.reshape((L1, 1)).repeat(
        (1, L2)) + normsqrY.reshape((1, L2)).repeat((L1, 1)) - 2 * inner_products

    # L1*L2
    distsqr0 = distsqr.reshape((L1*L2))

    if maxnum is not None and L2 > maxnum:
        distsqr_sorted = torch.sort(distsqr, dim=1)[0]
        cutoffs = distsqr_sorted[:, maxnum].reshape(L1, 1).repeat((1, L2))
        cutoffs0 = cutoffs.reshape((L1*L2))
        # just needs to be greater than sigma^2
        distsqr0[torch.nonzero(distsqr0 >= cutoffs0)] = sigma**2 + 1

    # L1*L2
    weights0 = torch.zeros((L1*L2), dtype=torch.float32, device=device)
    # todo: try it without inds_low
    inds_low = torch.nonzero(distsqr0 <= sigma**2).flatten()
    # inds_high = torch.nonzero(distsqr0>sigma**2).flatten()
    if len(inds_low) > 0:
        # weights0[inds_low] = 1
        weights0[inds_low] = torch.exp(-distsqr0[inds_low]/(sigma)**2)
    # L1 x L2
    weights = weights0.reshape((L1, L2))

    # Now accumulate the clips
    # L1 x M x T
    numer = torch.mm(weights, Y_t.reshape((L2, M*T))).reshape((L1, M, T))
    denom = torch.sum(weights, dim=[1])

    return numer, denom


def extract_clips_t(traces_t: torch.tensor, *, t_start: int, T_delta: int, T: int, num_clips: int) -> torch.tensor:
    """Extract clips from a recording, operating on torch tensors, with a given clip size and an increment between clips

    Parameters
    ----------
    traces_t : torch.tensor
        The recording tensor (M x T)
    t_start : int
        Start timepoint
    T_delta : int
        Increment between clips (in timepoints)
    T : int
        The clip (or snippet) size
    num_clips : int
        Number of clips to extract

    Returns
    -------
    torch.tensor
        The extracted clips (num_clips x M x T)
    """
    # traces_t: M x N
    # output: clips: num_clips x M x T
    L = num_clips
    M = traces_t.shape[0]
    assert T % T_delta == 0, f'T must be divisible by T_delta: T={T} T_delta={T_delta}'
    aa = int(T/T_delta)
    assert traces_t.shape[1] >= t_start + T_delta*(
        L+aa-1), f'traces_t not large enough: {traces_t.shape[1]} < {t_start + T_delta*(L+aa-1)}'

    # L+aa-1 x M x T_delta
    clipsA_t = traces_t[:, t_start:t_start + T_delta *
                        (L+aa-1)].reshape((M, L+aa-1, T_delta)).permute((1, 0, 2))
    # L x M x T
    clips_t = torch.zeros((L, M, T), dtype=torch.float32,
                          device=traces_t.device)
    for ii in range(aa):
        clips_t[:, :, T_delta*ii:T_delta*(ii+1)] = clipsA_t[ii:ii+L, :, :]
    return clips_t


def _get_neighborhoods(*, recording: se.RecordingExtractor, opts: EphysNlmV1Opts) -> List[Dict]:
    """Get a list of neighborhoods from a recording extractor based on the ephys_nlm options

    Parameters
    ----------
    recording : se.RecordingExtractor
        Recording extractor (SpikeInterface)
    opts : EphysNlmV1Opts
        Denoising options

    Returns
    -------
    List[Dict]
        List of dictionaries representing neighborhoods. Each dictionary contains information about the neighborhood, such as number of channels.
    """
    M = len(recording.get_channel_ids())
    if opts.multi_neighborhood is False:
        # A single neighborhood
        return [
            dict(
                channel_indices=np.arange(M),
                target_indices=np.arange(M)
            )
        ]
    geom: np.ndarray = _get_geom_from_recording(recording=recording)
    adjacency_radius = opts.neighborhood_adjacency_radius
    assert adjacency_radius is not None, 'You need to provide neighborhood_adjacency_radius when multi_neighborhood is True'
    ret = []
    for m in range(M):
        channel_indices = _get_channel_neighborhood(
            m=m, geom=geom, adjacency_radius=adjacency_radius)
        ret.append(dict(
            channel_indices=channel_indices,
            target_indices=[m]
        ))
    return ret


def _get_channel_neighborhood(m: int, geom: np.ndarray, *, adjacency_radius: float) -> np.ndarray:
    """Helper function used by _get_neighborhoods

    Parameters
    ----------
    m : int
        The index of the central channel of the neighborhood
    geom : np.ndarray
        The geom array [M x D] where the dimension D is either 2 or 3
    adjacency_radius : float
        The adjacency radius for determining the channels

    Returns
    -------
    np.ndarray
        Array of indices of the channels in the neighborhood
    """
    M = geom.shape[0]
    if adjacency_radius < 0:
        return np.arange(M)
    deltas = geom-np.tile(geom[m, :], (M, 1))
    distsqrs = np.sum(deltas**2, axis=1)
    inds = np.where(distsqrs <= adjacency_radius**2)[0]
    return inds.ravel()


def _get_geom_from_recording(recording: se.RecordingExtractor) -> np.ndarray:
    """Retrieve the electrode locations for a recording

    Parameters
    ----------
    recording : se.RecordingExtractor
        Recording extractor (SpikeInterface)

    Returns
    -------
    np.ndarray
        Geom array (M x D) where the dimension D is either 2 or 3
    """
    channel_ids = recording.get_channel_ids()
    M = len(channel_ids)
    location0 = recording.get_channel_property(channel_ids[0], 'location')
    nd = len(location0)
    geom = np.zeros((M, nd))
    for ch_id, ii in enumerate(channel_ids):
        location_ii = recording.get_channel_property(
            ch_id, 'location')
        geom[ii, :] = list(location_ii)
    return geom
