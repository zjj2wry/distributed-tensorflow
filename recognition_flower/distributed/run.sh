#!/bin/bash

export TF_CONFIG='{"cluster": {"ps": ["localhost:2223"], "worker": ["localhost:2222"]}, "task": {"index": 0, "type": "worker"}}'
python train_dataset.py &
export TF_CONFIG='{"cluster": {"ps": ["localhost:2223"], "worker": ["localhost:2222"]}, "task": {"index": 0, "type": "worker"}}'
python train_dataset.py &
