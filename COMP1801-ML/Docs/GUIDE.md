# Getting Started Guide

To create a virtual environment for the `COMP1801` course, run the following command:

```bash
conda create --name COMP1801-ML python=3.10.10
```

After creating the environment, activate it using:

```bash
conda activate COMP1801-ML
```

To install the required packages, navigate to the [Docs/](/Docs/) directory and run the following command:

```bash
pip install -r requirements.txt
```

To create GPU-enabled kernel, you can create one using the following command:

```bash
python -m ipykernel install --user --name COMP1801-ML --display-name="COMP1801-ML(GPU)"
```

For more information on creating a GPU-enabled kernel, refer to my [Notion documentation](https://muhammedazhar.notion.site/How-to-setup-a-conda-environment-e27054b819cd4864afd600886b768888?pvs=4) about creating a GPU-enabled kernel.
