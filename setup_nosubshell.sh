#!/bin/bash

# it looks like thisbecause it's supposed to run in a container
export WORK_DIR="/workspace/work/GuNFlows/"
export INSTALL_DIR="$WORK_DIR/software/install/"
export BUILD_DIR="$WORK_DIR/software/build/"
export REPO_DIR="$WORK_DIR/software/repo/"

export GUNDAM_HOME="$INSTALL_DIR/gundam"
export PATH=$GUNDAM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUNDAM_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/work/GuNFlows/src/gunflows:$GUNDAM_HOME/lib:$PYTHONPATH

# Path to datasets
export OA_INPUT_FOLDER="/workspace/data"
export DATASET_FOLDER="/workspace/data"
# path to config folder
export CONFIG_FOLDER="/workspace/config/GundamWorkspace/"
export GUNFLOW_SRC="/home/shares/sanchezf/gundam_n_flow/GuNFlows/src/gunflows"
export DATASET_FOLDER="/workspace/data"

source /opt/root/bin/thisroot.sh
echo "Environment variables set for Gundam and ROOT."

#echo "Starting Interactive shell..."
#exec bash
