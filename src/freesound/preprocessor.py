import os
from shutil import copyfile
from pathlib import Path
from freesound.utils.general import save_yaml, load_yaml, hash_dict, subdir_from_wavname
from freesound.data_utils.audio_transforms import path_to_im_fns
import pickle
import bz2
from fastprogress import progress_bar
import time

DEFAULT_CONFIG_PATH = 'config/preprocessing/default.yaml'
KAGGLE_INPUTS_DIR = os.environ['FS_INPUTS_BASE']
CACHE_STORE_DIR = Path(os.environ['FS_DATA_STORE_DIR']) / 'preprocessed_data'


class Preprocessor:
    def __init__(self, config_path=None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = load_yaml(self.config_path)
        self.config_hash = hash_dict(self.config)
        self.fs_data_dir = Path(KAGGLE_INPUTS_DIR) / 'freesound-audio-tagging-2019/'
        self.path_to_im_fn = path_to_im_fns[self.config['path_to_im_fn_name']]
        self.load_cache()

    def load_cache(self):
        self.cache_path = Path(CACHE_STORE_DIR) / self.config_hash
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.cache_pkl_path = self.cache_path / 'cache.pkl'
        if not os.path.exists(self.cache_pkl_path):
            with bz2.BZ2File(self.cache_pkl_path, 'w') as f:
                pickle.dump({}, f)
                save_yaml(self.config, self.cache_path / 'config.yaml')
        with bz2.BZ2File(self.cache_pkl_path, 'r') as f:
            now = time.time()
            self.cache = pickle.load(f)
            print('Loading took {} seconds'.format(time.time() - now))

    def save_cache(self):
        with bz2.BZ2File(self.cache_pkl_path, 'w') as f:
            pickle.dump(self.cache, f)

    def fill_cache(self, paths):
        for p in progress_bar(paths):
            self[p]

    def _save_config(self):
        spl = os.path.split(self.configconfig_path)
        self.save_config_path = Path(spl[0]) / 'cfg_images_hashes'
        if not self.save_config_path.exists():
            self.save_config_path.mkdir()
        copyfile(self.config_path, self.save_config_path / (self.config_hash + '_' + spl[1]))

    def image_from_wavhash(self, wav):
        subdir = subdir_from_wavname(wav)
        wavpath = self.fs_data_dir / subdir / wav
        return self.path_to_im_fn(wavpath, self.config['params'])

    def _clean_wavname(self, wavname):
        if '.' in wavname:  # rm file extension if any
            wavname = wavname.split('.')[0]
        return wavname + '.wav'

    def __getitem__(self, wavname):
        p = self._clean_wavname(wavname)
        if p not in self.cache:
            self.cache[p] = self.image_from_wavhash(p)
        return self.cache[p]
