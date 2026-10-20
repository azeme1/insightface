"""MobileFaceNet + ArcFace on MS1MV3, single device (GPU or CPU).

get_config() merges this on top of configs/base.py, so only fields that differ
from base.py are set here (base.py is an r50 skeleton missing the dataset keys).
Same recipe as upstream configs/ms1mv3_mbf.py; only rec/output/workers differ.
"""
from easydict import EasyDict as edict

config = edict()

# model (base.py defaults to r50 / fp32)
config.network = "mbf"
config.fp16 = True
config.weight_decay = 1e-4

# data (absent from base.py)
config.rec = "/workspace/home_0/dataset/faces/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

# run (base.py: output=r50 dir, workers=2, no full-state checkpoints)
config.output = "work_dirs/ms1mv3_mbf"
config.num_workers = 16
config.save_all_states = True    # full state each epoch so the run can be resumed
