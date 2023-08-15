# NVIDIA CUDA DevContainer Template with GPU Support on Windows

Build and run a Ubuntu-based container with CUDA GPU support on Windows to avoid platform-related compatibility and update issues, which also supports running `.py` and `.ipynb` scripts.

## Prerequisites

* [Docker engine](https://docs.docker.com/engine/install/) or [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) (and setup [.wslconfig](https://learn.microsoft.com/en-us/windows/wsl/wsl-config) to use more cores and memory than default)
* [NVIDIA driver](https://www.nvidia.com/download/index.aspx) for the graphic card
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (which is already included in Windows’ Docker Desktop)
* [VS Code](https://code.visualstudio.com/download) with [DevContainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed

See [here](https://www.tensorflow.org/install/pip#hardware_requirements) for hardware and system requirements of running Tensorflow.

## Start DevContainer

Modify [requirements.txt](https://github.com/alankrantas/windows-cuda-gpu-devcontainer/blob/main/.devcontainer/requirements.txt) to include packages you'd like to install. `ipykernel` is required for executing IPython notebook cells in VS Code.

Open the folder in VS Code, press `F1` to bring up the Command Palette, and select **Dev Containers: Open Folder in Container...**.

After the DevContainer is up and running, test the GPU support with

```python
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Test run the example file:

```python
python3 autokeras-test.py
```

## Resources

* [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers)
* [NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
* [Setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras on Windows](https://alankrantas.medium.com/setup-a-nvidia-devcontainer-with-gpu-support-for-tensorflow-keras-on-windows-d00e6e204630)

See [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install) for the latest version of `libcudnn8` and `libcudnn8-dev` in [install-dev-tools.sh](https://github.com/alankrantas/windows-cuda-gpu-devcontainer/blob/main/.devcontainer/install-dev-tools.sh).
