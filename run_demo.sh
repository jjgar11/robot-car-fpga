#!/usr/bin/env bash
set -e

source /etc/profile.d/xrt_setup.sh

cd /home/xilinx/nano_gtsrb_robot_demo

exec /usr/local/share/pynq-venv/bin/python3 \
    robot_main.py \
    --full \
    --confirm
