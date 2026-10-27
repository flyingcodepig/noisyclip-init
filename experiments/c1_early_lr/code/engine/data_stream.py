from raw_cache import load_cache
import numpy as np
import torch
from src.data.image_cache import CachedImageDataset, collate_cached

class StreamDataset(CachedImageDataset):

    def __init__(self, *args, stream_states=None, stream_offset=0, workers=8, alternate_cache=None, alternate_probability=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_states = stream_states or {}
        self.stream_offset, self.workers = (stream_offset, workers)
        self.alternate = load_cache(alternate_cache) if alternate_cache else None
        self.alternate_probability = alternate_probability
        assert self.alternate is None or self.alternate.shape == self.arr.shape

    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        number = ((worker.id if worker else 0) + self.stream_offset) % max(1, self.workers)
        if self._rng is None:
            self._rng = np.random.default_rng(self.base_seed + 104729 * number)
            if number in self.stream_states:
                self._rng.bit_generator.state = self.stream_states[number]
        primary = self.arr
        if self.alternate is not None and self._rng.random() < self.alternate_probability:
            self.arr = self.alternate
        try:
            if self._randaug is not None:
                seed = int(self._rng.integers(0, 2 ** 31))
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    value = super().__getitem__(index)
            else:
                value = super().__getitem__(index)
        finally:
            self.arr = primary
        value['stream_id'] = number
        value['stream_state'] = self._rng.bit_generator.state
        return value

def stream_collate(batch):
    value = collate_cached(batch)
    value['stream_states'] = {sample['stream_id']: sample['stream_state'] for sample in batch}
    return value
