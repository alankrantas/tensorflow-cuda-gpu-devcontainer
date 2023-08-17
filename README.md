# NVIDIA CUDA + cuDNN DevContainer Template with GPU Support

Build and run a DevContainer with Python 3, CUDA 11.8 and cuDNN. This is a better way to run Tensorflow/AutoKeras on Windows with GPU support without frustrating installation and compatibility issues. `.py` and `.ipynb` scripts are supported without the need to install Anaconda/Jupyter Notebook.

## Prerequisites

* An amd64 (x64) machine with a CUDA-compatible NVIDIA graphic card
* [Docker engine](https://docs.docker.com/engine/install/) or [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) (and setup [.wslconfig](https://learn.microsoft.com/en-us/windows/wsl/wsl-config) to use more cores and memory than default if you are on Windows.)
* [NVIDIA graphic card driver](https://www.nvidia.com/download/index.aspx)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (which is already included in Windows’ Docker Desktop)
* [Visual Studio Code](https://code.visualstudio.com/download) with [DevContainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed

See [here](https://www.tensorflow.org/install/pip#hardware_requirements) for more detailed hardware and system requirements of running Tensorflow.

Be warned that some deep learning models require more GPU memory than others and may cause the Python kernel to crash. You may need to set a smaller batch for training.

## Start DevContainer

Modify [requirements.txt](https://github.com/alankrantas/windows-cuda-gpu-devcontainer/blob/main/.devcontainer/requirements.txt) to include packages you'd like to install. `ipykernel` is required for executing IPython notebook cells in VS Code.

Open the folder in VS Code, press `F1` to bring up the Command Palette, and select **Dev Containers: Open Folder in Container...**

Wait until the DevContainer is up and running, then test if the Tensorflow can detect the GPU correctly:

```python
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Test run using the example file:

```python
python3 autokeras-test.py
```

Or open `autokeras-test.ipynb` and run the cells.

After that, simply start Docker then open the directory in VS Code to use the built container.

## Resources

* [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers)
* [NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
* [Setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras on Windows](https://alankrantas.medium.com/setup-a-nvidia-devcontainer-with-gpu-support-for-tensorflow-keras-on-windows-d00e6e204630)

See [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install) for the latest version of `libcudnn8` and `libcudnn8-dev` in [install-dev-tools.sh](https://github.com/alankrantas/windows-cuda-gpu-devcontainer/blob/main/.devcontainer/install-dev-tools.sh).
