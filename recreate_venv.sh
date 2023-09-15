VENVS_DIR=${HOME}/.venvs
mkdir -p ${VENVS_DIR}

virtualenv --clear -p python3.7 ${VENVS_DIR}/cless_venv
source ${VENVS_DIR}/cless_venv/bin/activate

# one of subpackage has deprecated dependencies
# The 'sklearn' PyPI package is deprecated, use 'scikit-learn' rather than 'sklearn' for pip commands.
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

pip install -r requirements.txt
echo "export PYTHONPATH=`pwd`/cless" >> $VIRTUAL_ENV/bin/activate
