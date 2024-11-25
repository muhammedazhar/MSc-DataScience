# `COMP1801` - Machine Learning Coursework

## Description

This repository contains the machine learning coursework for COMP1801, focused on predicting the lifespan of metal parts and identifying whether they are defective. The coursework is divided into two main tasks: **Regression** to predict the lifespan of parts, and **Classification** to determine whether a part meets the company's quality threshold. The main goal is to help the company efficiently evaluate parts using machine learning models instead of slow, destructive testing methods.

## Repository Structure

```bash
.
├── Datasets
│   └── Dataset.csv
├── Docs
│   ├── Links.md
│   └── requirements.txt
├── Models
│   └── README.md
├── README.md
├── Reports
│   ├── Coursework.sty
│   ├── Images
│   │   ├── ClassDistribution.png
│   │   ├── DataVisualisationScatterPlot.png
│   │   ├── FeatureCrafting-KMeanClustering.png
│   │   ├── FeatureImportance-NeuralNetwork.png
│   │   ├── NeuralNetworkModel-TestSet-1.png
│   │   ├── NeuralNetworkModel-TestSet-2.png
│   │   └── PartialDependence-Top3.png
│   ├── References.bib
│   ├── Report.pdf
│   ├── Report.tex
│   └── fancyhdr.sty
├── Solutions
│   ├── Notebook.ipynb
│   └── Notebook.pdf
└── Testing
    ├── Encoding.py
    ├── T1-HyperTuner-NN.py
    ├── T1-HyperTuner-RF.py
    ├── T1-HyperTuner-XGB.py
    ├── T1-LinearRegression.ipynb
    ├── T1-NeuralNet.ipynb
    ├── T1-NeuralNet.py
    ├── T1-RandomForest.py
    ├── T1-Regression.ipynb
    ├── T1-XGBoost.py
    └── T2-Classification.ipynb
```

### Folder Descriptions

- **Datasets**: Contains the dataset (`Dataset.csv`) used for the entire coursework.
- **Docs**: Includes `requirements.txt` with the packages required to run the code, and `Links.md` with useful resources.
- **Models**: This folder is intended for storing the trained models and related descriptions.
- **Reports**: Contains the final coursework report in PDF (`Report.pdf`) and LaTeX (`Report.tex`) formats, along with supporting files and images used in the report.
- **Solutions**: Stores the main Jupyter Notebook (`Notebook.ipynb`) containing the integrated solution for the coursework, as well as a PDF version.
- **Testing**: Contains various Python scripts and Jupyter Notebooks used for testing, hyperparameter tuning, and validating different models (e.g., encoding schemes, hyperparameter tuning for Neural Networks, Random Forests, and XGBoost).

## Installation and Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/muhammedazhar/MSc-DataScience
   cd MSc-DataScience/COMP1801-ML/Coursework/
   ```

2. Install the required dependencies:

   ```bash
   cd Docs
   pip install -r Docs/requirements.txt
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open the `Solutions/Notebook.ipynb` file to explore the solution.

## Methods Overview

- **Regression**: Implemented using models such as Linear Regression, Random Forest, and an Artificial Neural Network (ANN). The ANN was chosen due to its robust ability to model complex relationships in the dataset.
- **Classification**: The Logistic Regression and ANN models were used to determine whether a part is defective, focusing on labeling parts that last less than 1500 hours. The ANN was also preferred here for its improved handling of minority classes.

## Key Findings

- The **Artificial Neural Network** model outperformed the others in both regression and classification tasks, demonstrating reliability, better feature capturing, and effective handling of imbalanced data.
- **Feature Crafting**: Used K-Means clustering for creating non-binary labels to distinguish higher-quality parts.
- **Evaluation Metrics**: RMSE, R², Weighted F1-Score, and Recall metrics were used to evaluate model performance and determine the most suitable approach.

## How to Run Tests

The **Testing** folder contains all scripts used for model testing, encoding evaluation, and hyperparameter tuning. Key steps:

1. Navigate to the `Testing` directory.
2. Run any of the `.py` or `.ipynb` files to experiment with different models and parameters.

Example:

```bash
# Show help
python HyperTuner-XGB.py -h

# Run with specific arguments
python HyperTuner-XGB.py --method random --iterations 100 --seed 42

# Run with grid search
python HyperTuner-XGB.py --method grid

# Run with verbose output
python HyperTuner-XGB.py --method random --verbose

# Run with custom data path
python HyperTuner-XGB.py --method random --data "/path/to/data.csv"

# Show help
python HyperTuner-RF.py -h

# Run with random search
python HyperTuner-RF.py --method random --iterations 100

# Run with grid search
python HyperTuner-RF.py --method grid --verbose

# Show help
python HyperTuner-NN.py -h

# Run with random search
python HyperTuner-NN.py --method random --iterations 100

# Run with grid search
python HyperTuner-NN.py --method grid --verbose
```

This script runs the hyperparameter tuning for the Neural Network.

## Report and Documentation

- The final report is available in the **Reports** folder as `Report.pdf`. This document includes data exploration, feature crafting, model selection, evaluation, and conclusions.
- All LaTeX sources for the report are also included for anyone interested in editing or generating an updated version.

## Acknowledgements

- **Module Leaders**: Dr. Peter Soar and Dr. Ilya Alexakhin for guidance throughout the course.
- **Resources**: Lecture materials and various resources linked in `Docs/Links.md` were used extensively.

## License

This project is part of the COMP1801 Machine Learning coursework at the University of Greenwich and should not be used for any commercial purposes or copied for academic misconduct.

## Contact

For any questions or clarifications, please reach out to me at <am7759c@gre.ac.uk>.

---
