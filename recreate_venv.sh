VENVS_DIR=${HOME}/.venvs
mkdir -p ${VENVS_DIR}

virtualenv --clear -p python3.7 ${VENVS_DIR}/cless_venv
source ${VENVS_DIR}/cless_venv/bin/activate

pip install -r requirements.txt
echo "export PYTHONPATH=`pwd`" >> $VIRTUAL_ENV/bin/activate
