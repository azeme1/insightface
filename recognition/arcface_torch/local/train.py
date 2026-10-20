"""Run arcface_torch/train_v2.py with a config kept outside the configs/ package.

train_v2.py's get_config() only imports from its own `configs` package, so this
registers the given file as `configs.<name>` first, then runs train_v2.py.

Env: /venv/arcface (real mxnet 1.9.1, numpy 1.23.5, torch 2.7.1+cu128).
Usage (from arcface_torch/):
  /venv/arcface/bin/python local/train.py local/configs/ms1mv3_mbf.py
"""
import argparse
import importlib
import importlib.util
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARCFACE = os.path.normpath(os.path.join(HERE, ".."))

parser = argparse.ArgumentParser()
parser.add_argument("config", help="path to a config .py defining `config`")
args = parser.parse_args()

sys.dont_write_bytecode = True  # keep the clone free of __pycache__
sys.path.insert(0, ARCFACE)

name = os.path.splitext(os.path.basename(args.config))[0]
importlib.import_module("configs")
spec = importlib.util.spec_from_file_location(f"configs.{name}", os.path.abspath(args.config))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules[f"configs.{name}"] = module

sys.argv = [os.path.join(ARCFACE, "train_v2.py"), f"configs/{name}.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
