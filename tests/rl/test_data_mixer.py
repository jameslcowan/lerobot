# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for RL data mixing (DataMixer, OnlineOfflineMixer)."""

import torch

from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources import OnlineOfflineMixer
from lerobot.utils.constants import OBS_STATE


def _make_buffer(capacity: int = 100, state_dim: int = 4) -> ReplayBuffer:
    buf = ReplayBuffer(
        capacity=capacity,
        device="cpu",
        state_keys=[OBS_STATE],
        storage_device="cpu",
        use_drq=False,
    )
    for i in range(capacity):
        buf.add(
            state={OBS_STATE: torch.randn(state_dim)},
            action=torch.randn(2),
            reward=1.0,
            next_state={OBS_STATE: torch.randn(state_dim)},
            done=bool(i % 10 == 9),
            truncated=False,
        )
    return buf


def test_online_only_mixer_sample():
    """OnlineOfflineMixer with no offline buffer returns online-only batches."""
    buf = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(online_buffer=buf, offline_buffer=None, online_ratio=0.5)
    batch = mixer.sample(batch_size=8)
    assert batch["state"][OBS_STATE].shape[0] == 8
    assert batch["action"].shape[0] == 8
    assert batch["reward"].shape[0] == 8


def test_online_only_mixer_ratio_one():
    """OnlineOfflineMixer with online_ratio=1.0 and no offline is equivalent to online-only."""
    buf = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(online_buffer=buf, offline_buffer=None, online_ratio=1.0)
    batch = mixer.sample(batch_size=10)
    assert batch["state"][OBS_STATE].shape[0] == 10


def test_online_offline_mixer_sample():
    """OnlineOfflineMixer with two buffers returns concatenated batches."""
    online = _make_buffer(capacity=50)
    offline = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(
        online_buffer=online,
        offline_buffer=offline,
        online_ratio=0.5,
    )
    batch = mixer.sample(batch_size=10)
    assert batch["state"][OBS_STATE].shape[0] == 10
    assert batch["action"].shape[0] == 10
    # 5 from online, 5 from offline (approx)
    assert batch["reward"].shape[0] == 10


def test_online_offline_mixer_iterator():
    """get_iterator yields batches of the requested size."""
    buf = _make_buffer(capacity=50)
    mixer = OnlineOfflineMixer(online_buffer=buf, offline_buffer=None)
    it = mixer.get_iterator(batch_size=4, async_prefetch=False)
    batch1 = next(it)
    batch2 = next(it)
    assert batch1["state"][OBS_STATE].shape[0] == 4
    assert batch2["state"][OBS_STATE].shape[0] == 4


def test_replay_buffer_accepts_passthrough_sampling_kwargs():
    """ReplayBuffer ignores unknown kwargs so future algos can pass hints safely."""
    buf = _make_buffer(capacity=20)
    _ = buf.sample(batch_size=4, n_steps=3, gamma=0.99, action_chunk_size=8)
    it = buf.get_iterator(
        batch_size=4,
        async_prefetch=False,
        n_steps=3,
        gamma=0.99,
        action_chunk_size=8,
    )
    _ = next(it)


def test_mixer_forwards_kwargs_in_all_paths():
    """OnlineOfflineMixer should forward kwargs in online-only and mixed modes."""

    class _StubBuffer:
        def __init__(self):
            self.sample_calls = []
            self.iterator_calls = []

        def sample(self, batch_size, **kwargs):
            self.sample_calls.append((batch_size, kwargs))
            return {
                "state": {OBS_STATE: torch.randn(batch_size, 4)},
                "action": torch.randn(batch_size, 2),
                "reward": torch.randn(batch_size),
                "next_state": {OBS_STATE: torch.randn(batch_size, 4)},
                "done": torch.zeros(batch_size),
                "truncated": torch.zeros(batch_size),
                "complementary_info": None,
            }

        def get_iterator(self, batch_size, async_prefetch=True, queue_size=2, **kwargs):
            self.iterator_calls.append((batch_size, async_prefetch, queue_size, kwargs))
            while True:
                yield self.sample(batch_size, **kwargs)

    online = _StubBuffer()
    offline = _StubBuffer()

    # Online-only path delegates to online buffer iterator with kwargs.
    online_only = OnlineOfflineMixer(online_buffer=online, offline_buffer=None, online_ratio=1.0)
    it = online_only.get_iterator(batch_size=4, async_prefetch=False, n_steps=5, gamma=0.95)
    _ = next(it)
    assert online.iterator_calls
    assert online.iterator_calls[-1][3] == {"n_steps": 5, "gamma": 0.95}

    # Mixed path should forward kwargs to both sample() calls.
    mixed = OnlineOfflineMixer(online_buffer=online, offline_buffer=offline, online_ratio=0.5)
    _ = mixed.sample(batch_size=6, n_steps=4, action_chunk_size=12)
    assert online.sample_calls[-1][1] == {"n_steps": 4, "action_chunk_size": 12}
    assert offline.sample_calls[-1][1] == {"n_steps": 4, "action_chunk_size": 12}
