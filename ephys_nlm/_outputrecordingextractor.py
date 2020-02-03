import numpy as np
import spikeextractors as se

class OutputRecordingExtractor(se.RecordingExtractor):
    def __init__(self, *, base_recording, block_size):
        super().__init__()
        self._base_recording = base_recording
        self._block_size = block_size
        self.copy_channel_properties(recording=self._base_recording)

        self._blocks = []

    def add_block(self, traces):
        if traces.shape[1] == self._block_size:
            self._blocks.append(traces)
        else:
            if traces.shape[1] >= self._block_size * 2:
                raise Exception(
                    'Unexpected error adding block to OutputRecordingExtractor.') # pragma: no cover
            if len(self._blocks) * self._block_size + traces.shape[1] != self.get_num_frames():
                raise Exception(f'Problem adding final block. Unexpected size: {traces.shape[1]}. Block size is {self._block_size}. Number of frames is {self.get_num_frames()}.') # pragma: no cover
            if traces.shape[1] > self._block_size:
                self._blocks.append(traces[:, :self._block_size])
                self._blocks.append(traces[:, self._block_size:])
            else:
                self._blocks.append(traces)

    def get_channel_ids(self):
        return self._base_recording.get_channel_ids()

    def get_num_frames(self):
        return self._base_recording.get_num_frames()

    def get_sampling_frequency(self):
        return self._base_recording.get_sampling_frequency()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()

        channel_indices = []
        aa = self.get_channel_ids()
        for ch in channel_ids:
            channel_indices.append(np.nonzero(np.array(aa) == ch)[0][0])

        ib1 = int(start_frame / self._block_size)
        ib2 = int((end_frame-1) / self._block_size)

        assert ib2 < len(self._blocks), f'Block {ib2} not found in OutputRecordingExtractor (num blocks is {len(self._blocks)})'

        trace_blocks = []
        if ib1 == ib2:
            trace_blocks.append(
                self._blocks[ib1][channel_indices][:, start_frame - ib1 * self._block_size:end_frame - ib1 * self._block_size]
            )
        else:
            trace_blocks.append(
                self._blocks[ib1][channel_indices][:,
                                                   start_frame - ib1 * self._block_size:]
            )
            for ii in range(ib1 + 1, ib2):
                trace_blocks.append(
                    self._blocks[ii][channel_indices, :]
                )
            trace_blocks.append(
                self._blocks[ib2][channel_indices][:,
                                                   :end_frame - ib2 * self._block_size]
            )
        return np.concatenate(trace_blocks, axis=1)
