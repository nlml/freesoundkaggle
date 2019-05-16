import numpy as np
import torch
import yaml
import os
import hashlib
import tensorboardX
from pathlib import Path
from shutil import copyfile


def str_repr(d):
    s = ''
    for k in sorted(list(d.keys())):
        if type(d[k]) is dict:
            s += '' + str(k) + ': {' + str_repr(d[k]) + '}, '
        else:
            s += '' + str(k) + ': ' + str(d[k]) + ', '
    return s


def hash_dict(config, limit_chars=16):
    hash_object = hashlib.sha256(str_repr(config).encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    return hex_dig[:limit_chars]


def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    return config


def save_yaml(data, path):
    with open(path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_tboard_writer(config_hash):
    tboard_log_dir = 'tboard/' + config_hash
    tboard_writer = tensorboardX.SummaryWriter(tboard_log_dir)
    return tboard_writer, Path(tboard_log_dir)


def load_config_and_setup_tboard(config_filepath):
    config_filepath = Path(config_filepath)
    config = load_yaml(config_filepath)
    config_hash = hash_dict(config)
    tboard_writer, tboard_log_dir = setup_tboard_writer(config_hash)
    config_save_filepath = tboard_log_dir / os.path.split(config_filepath)[1]
    copyfile(config_filepath, config_save_filepath)
    return config, config_hash, tboard_writer


def load_config_from_hash(hsh, dir):
    config_path = Path(dir)
    if hsh is not None:
        ls = [i for i in os.listdir(config_path) if hsh in i]
    else:
        ls = [i for i in os.listdir(config_path) if '.yaml' in i]
    config_path = config_path / ls[0]
    config = load_yaml(config_path)
    return config


def subdir_from_wavname(wav, fs_data_dir=None):
    fs_data_dir = fs_data_dir or (
        Path(os.environ['FS_INPUTS_BASE']) / 'freesound-audio-tagging-2019/')
    fs_data_dir = Path(fs_data_dir)
    for subdir in ['train_curated', 'test', 'train_noisy']:
        wavpath = fs_data_dir / subdir / wav
        if wavpath.exists():
            return subdir
    assert 0, 'shouldnt be here!'
