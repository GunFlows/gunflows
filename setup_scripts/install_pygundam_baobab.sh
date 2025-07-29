# GUNDAM software directory structure
export WORK_DIR="/home/shares/sanchezf/gundam_n_flow/GuNFlows/software"
export INSTALL_DIR="$WORK_DIR/install/"
export BUILD_DIR="$WORK_DIR/build/"
export REPO_DIR="$WORK_DIR/repo/"

# Path to datasets
export OA_INPUT_FOLDER="/srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud/"
# path to config folder
export CONFIG_FOLDER="/srv/beegfs/scratch/groups/dpnc/neutrinos/GundamInputOA2021/"

# Load venv (Python3.10)
# check python version
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1-2)
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "Error: Python version is not 3.10. Please use Python 3.10."
    exit 1
fi

##################################
#  YOU NEED PYBIND!!
##################################
if ! python3 -c "import pybind11" &> /dev/null; then
    echo "Error: pybind11 is not installed." >&2
    return 1 2>/dev/null || exit 1
fi

# Build Gundam with Python bindings
HERE=$(pwd)
cd $BUILD_DIR/gundam
cmake \
  -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/gundam \
  -D CMAKE_BUILD_TYPE=Release \
  -D WITH_PYTHON_INTERFACE=ON \
  $REPO_DIR/gundam/.
  make install
cd $HERE

export PYTHONPATH="$INSTALL_DIR/gundam/lib:$PYTHONPATH"

# test
python3 -c "import GUNDAM"
python3 -c "import ROOT"
