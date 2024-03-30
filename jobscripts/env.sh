module purge
module load Python/3.11.3-GCCcore-12.3.0

python3 -m venv $HOME/venvs/nlp_fp

source $HOME/venvs/nlp_fp/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

python3 --version