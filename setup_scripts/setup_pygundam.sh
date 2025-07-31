# GUNDAM software directory structure
export WORK_DIR="$HOME/T2K-uniGe/Gundam"
export INSTALL_DIR="$WORK_DIR/install/"
export BUILD_DIR="$WORK_DIR/build/"
export REPO_DIR="$WORK_DIR/repo/"

# Path to datasets
export OA_INPUT_FOLDER="/Users/lorenzo/T2K-uniGe/Gundam/GundamInputOA2021/Datasets/"
export CONFIG_FOLDER="/Users/lorenzo/T2K-uniGe/Gundam/GundamInputOA2021/"

# Load venv (Python3.10)
source /Users/lorenzo/my-python-3.10-env/bin/activate

# build Gundam with python bindings (only once)
#cd "${BUILD_DIR}"/gundam cmake -D WITH_PYTHON_INTERFACE=ON ./  make install || exit


##################################
#  YOU NEED PYBIND!!
##################################
python3 -c "import pybind11; print(pybind11.get_include())"


export PYTHONPATH="$INSTALL_DIR/gundam/lib:$PYTHONPATH"

# test
python3 -c "import GUNDAM"
python3 -c "import ROOT"
