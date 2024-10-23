# Getting Started Guide

To create a virtual environment for the `STAT1050` course, run the following command:

```bash
conda create --name STAT1050-SM python=3.10.10
```

After creating the environment, activate it using:

```bash
conda activate STAT1050-SM
```

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

Creating a virtual environment for kernel

```bash
python -m ipykernel install --user --name=STAT1050-SM --display-name="STAT1050-SM(IPYNB)"
```

For more information on creating a GPU-enabled kernel, refer to my [Notion documentation](https://muhammedazhar.notion.site/How-to-setup-a-conda-environment-e27054b819cd4864afd600886b768888?pvs=4) about creating a GPU-enabled kernel.

To check whether this machine has **R** installed, run the following command:

```bash
R --version
```

If you don't have **R** installed, follow the instructions on the [R Project website](https://www.r-project.org/) to install it.

In this subject we will be using Jupyter Notebook and we will also be running R inside the notebook. Therefore, we need to install the `IRkernel` package to access R inside the notebook. To do this, run the following steps:

1. Go to the R console with superuser access for system-wide installation by running `sudo R` command in the terminal.
2. Run the following command line by line in the R console:

    ```R
    install.packages('IRkernel')
    IRkernel::installspec(user = FALSE)
    ```

    The first line of command may prompt user to select a CRAN mirror, which is a server from which R will download the package. It is recommended to select a mirror close to your location. If you want it automatically routing to the nearest server, choose `1: 0-Cloud [https]` the first option.
    The second command will make the R kernel available to Jupyter. Using `user = FALSE` will install it system-wide, making it available to all users.

3. Exit the R console by running `q()` command and type `y` to save changes. After that, restart the Jupyter Notebook server and the R kernel should be available in drropdown list.

## Runing R scripts

Use the `Rscript` command along with `.R` file to run R scripts:

```bash
Rscript your_filename.R
```
