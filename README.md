# CSB320_P1

## Workflow

run: 
```
conda env create -f requirements.yml
conda activate csb320-env
```

Then simply run all Jupyter cells in diabetes_analyses.ipynb and cancer_analyses.ipynb

## File Descriptions

### lib.py

Imports all modules, defines some basic functions for use in library

### pipeline_presets.py

Contains presets for configurations of each learning model

### grid_search_presets

Contains presents for each parameter grid for each learning model

### diabetes_analyses.ipynb (and .html)

Evaluation and report of machine learning methods on Diabetes dataset

### cancer_analyses.ipynb (and .html)

Evaluation and report of machine learning methods on Wisconsin Breast Cancer dataset

### discussion_report.ipynb (and .html)

Discussions, conclusions, and reflections of results from two reports

