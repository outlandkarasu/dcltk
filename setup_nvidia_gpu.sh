#!/bin/sh

sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2505,875
sudo nvidia-smi --auto-boost-default=DISABLED

