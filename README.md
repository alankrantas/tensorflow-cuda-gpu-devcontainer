# Tensorflow CUDA DevContainer Configuration for Supporting NVIDIA GPU

To create a Docker Linux DevContainer that supports Tensorflow GPU without frustrating and complicated local installation, especially on Windows platforms.

The current version willl create the following setup:

* Python `3.11`
* Tensorflow `2.15.0`
* CUDA `12.2` + cuDNN

> The current version utilizes `tensorflow[and-cuda]` to install compatible CUDA/cuDNN on a regular Python container. My original and previous version used a NVIDIA CUDA image to install Python and cuDNN.
> 
> For now Tensorflow `2.16.1` cannot be installed currectly. And, I still have no success to have installed TensorRT detected. Let me know if you managed to get it running!

The current version has been tested on:

* A Windows 11 gaming laptop with a built-in RTX 3070 Ti

> I've tried with my another laptop connecting to a eGPU (GTX 1660 Ti) without success so far.

## Prerequisites

* An amd64 (x64) machine with a CUDA-compatible NVIDIA GPU card
* [Docker engine](https://docs.docker.com/engine/install/) or [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) (and setup [.wslconfig](https://learn.microsoft.com/en-us/windows/wsl/wsl-config) to use more cores and memory than default if you are on Windows.)
* Latest version of the [NVIDIA graphic card driver](https://www.nvidia.com/download/index.aspx)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (which is already included in Windows’ Docker Desktop)
* [Visual Studio Code](https://code.visualstudio.com/download) with [DevContainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed

See [here](https://www.tensorflow.org/install/pip#hardware_requirements) for more detailed hardware and system requirements of running Tensorflow.

> Note: some deep learning models require more GPU memory than others and may cause the Python kernel to crash. You may need to try setting a smaller training batch size.
>
> Some older cards may only support older drivers thus older CUDA versions. See the CUDA [Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html) for details, although I cannot test if `tensorflow[and-cuda]` will install older libraries automatically.

## Download Repo

Download the repo as a .zip file, unzip then open the folder in VS Code, or to use [Git](https://git-scm.com/):

```bash
git clone https://github.com/alankrantas/tensorflow-cuda-gpu-devcontainer.git
cd tensorflow-cuda-gpu-devcontainer
code .
```

Modify [requirements.txt](https://github.com/alankrantas/windows-cuda-gpu-devcontainer/blob/main/.devcontainer/requirements.txt) to include packages you'd like to install. `ipykernel` is required for executing IPython notebook cells in VS Code.

> Note: VS Code/Git may messed up the new line characters of `.devcontainer/install-dev-tools.sh` and `.devcontainer/requirements` which will cause the DevContainer failed to run the scripts.
>
> Click `CRLF` on the button right and change them to `LF`.

## Start DevContainer

In VS Code, press `F1` to bring up the Command Palette, and select **Dev Containers: Open Folder in Container...**

Wait until the DevContainer is up and running (`nvidia-smi` is executed):

```
Wed May  1 03:24:54 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.76.01              Driver Version: 552.22         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   46C    P0             29W /  130W |       0MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

> Although `nvidia-smi` informed us the current NVIDIA driver uses CUDA `12.4`, it doesn't mean Tensorflow supports it. This is why using `tensorflow[and-cuda]` is easier to use existing NVIDIA CUDA/cuDNN images.

Then open a new terminal (`Terminal` -> `New Terminal`) to test if the Tensorflow detects the GPU correctly:

```python
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If success, you should see something like

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Afterwards, simply open the folder in VS Code to restart the built container.

## Test Script

Test run using the example file - open `autokeras-test.py` and seletct `Run` -> `Run Without Debugging`, or run in the terminal

```python
python3 autokeras-test.py
```

Which generates a result like below:

```
Trial 1 Complete [00h 03m 51s]
val_loss: 0.03894084319472313

Best val_loss So Far: 0.03894084319472313
Total elapsed time: 00h 03m 51s

...

Prediction loss: 0.0315
Prediction accuracy: 0.9910
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       980
           1       1.00      1.00      1.00      1135
           2       0.99      0.99      0.99      1032
           3       0.99      1.00      0.99      1010
           4       0.99      0.99      0.99       982
           5       0.99      0.99      0.99       892
           6       1.00      0.98      0.99       958
           7       0.99      0.99      0.99      1028
           8       0.99      0.99      0.99       974
           9       0.99      0.99      0.99      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
```

Or open `autokeras-test.ipynb`, run the cells or select `Run All` on top of the notebook.

## Resources

* [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers)
* [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
* [Setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras on Windows](https://alankrantas.medium.com/setup-a-nvidia-devcontainer-with-gpu-support-for-tensorflow-keras-on-windows-d00e6e204630)
