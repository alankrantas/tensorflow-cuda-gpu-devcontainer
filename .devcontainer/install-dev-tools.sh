# update system
apt update
apt upgrade -y
# install Linux tools and Python 3
apt install -y software-properties-common wget python3-dev python3-pip python3-wheel python3-setuptools
# install Python packages
python3 -m pip install --upgrade pip
pip3 install --user --extra-index-url https://pypi.nvidia.com -r .devcontainer/requirements.txt
# clean up
pip3 cache purge
