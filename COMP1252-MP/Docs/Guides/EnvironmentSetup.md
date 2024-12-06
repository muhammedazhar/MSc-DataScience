# Environment Setup for the COMP1252-MP Project

## 1. Install Anaconda

Download the latest version of Anaconda from [here](https://www.anaconda.com/products/individual). You also other anaconda distributions like Miniconda.

## 2. Create a new environment

To create a new environment, run the following commands:

```bash
conda create --name COMP1252-MP python=3.10
conda activate COMP1252-MP
```

## 3. Install the required packages

To install the required packages, navigate to `Docs/` directory run the following command:

```bash
pip install -r requirements.txt
```

## 4. Install the required Jupyter GPU Kernel

To install the required Jupyter GPU Kernel, run the following command:

```bash
python -m ipykernel install --user --name=COMP1252-MP --display-name "COMP1252-MP(GPU)"
```

The confirm the installation of the Jupyter Kernel, check the available kernels in the Jupyter Notebook kernel selection menu or by running the following command:

```bash
jupyter kernelspec list
```

Alternatively, you can also run the [`kernel_check.py`](../../Solutions/kernel_check.py) script to check the installation of the Jupyter Kernel.
Look at the top-right corner of the notebook interface. The name of the currently selected kernel is displayed there.
