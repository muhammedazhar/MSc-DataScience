# `COMP1801` - Machine Learning Coursework

How to use the custom hyper parameter:

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
```
