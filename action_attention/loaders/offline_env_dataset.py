from pathlib import Path

import numpy as np
from PIL import Image

from .Dataset import Dataset


INFO_FILE = 'info.npy'


class OfflineEnvironmentDataset(Dataset):
    def __init__(self,
                 root: str,
                 path_length: int = None,
                 num_episodes: int = 10000,
                 num_episode_steps: int = 100,
                 clip_length: int = None,
                 kind='transition',
                 ) -> None:
        super().__init__()

        self.root = Path(root)
        self.path_length = path_length
        self.info_file_path = self.root / INFO_FILE
        self.num_episodes = num_episodes
        self.num_episode_steps = num_episode_steps
        self.clip_length = clip_length

        self.info = None
        self.kind = kind

        if self.kind == 'transition':
            self.num_steps = self.num_episodes * self.num_episode_steps
        elif self.kind == 'path':
            assert self.clip_length >= self.path_length, f'clip_length={self.clip_length}, path_length={self.path_length}'
            self.indexes_per_episode = self.num_episode_steps // self.clip_length
            self.num_steps = self.num_episodes * self.indexes_per_episode
        else:
            raise ValueError(f'kind={self.kind}, expected: kind in (transition, path)')

        self.load_dataset()

    def load_dataset(self):
        assert self.info_file_path.exists()
        self.info = np.load(self.info_file_path, allow_pickle=True).item()

    def load_npy(self, path: Path):
        return np.load(path, allow_pickle=True)

    def _idx_to_episode_step(self, index):
        episode_id = index // self.num_episodes
        step_id = index % self.num_episode_steps

        return episode_id, step_id

    def __getitem__(self, idx):
        if self.kind == 'transition':
            return self._get_transition(idx)
        elif self.kind == 'path':
            return self._get_episode(idx)
        else:
            raise ValueError(f'Unexpected kind={self.kind}')

    def _get_transition(self, index):
        episode_id, step_id = self._idx_to_episode_step(index)
        obs, next_obs = [self.preprocess_image(self.load_obs(obs)) for obs in self.info[episode_id]["obs"][step_id: step_id + 2]]
        action = self.load_npy(self.info[episode_id]['actions'])[step_id]

        return obs, action, next_obs

    def _get_episode(self, index):
        ep = index // self.indexes_per_episode
        start_step = (index % self.indexes_per_episode) * self.clip_length
        last_step = start_step + self.path_length
        observations = [self.preprocess_image(self.load_obs(obs)) for obs in self.info[ep]["obs"][start_step:last_step + 1]]
        actions = self.load_npy(self.info[index]['actions'])[start_step:last_step]

        return observations, (actions,)

    def load_obs(self, obs_path: Path):
        with Image.open(obs_path) as img:
            return np.array(img)
