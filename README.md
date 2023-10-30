# FW23_AICUP

Team member: 鄧昱辰、林冠妤、陳瑾叡、游孟純、劉子豪

## Module structure

```plaintext
- /data
    - /external_data
    - training_data.csv
    - pred.csv
- /src
    - __init__.py
    - predict.py
    - preproc.py
    - utils.py
    - building_dist.py
    - knn_test.py
    - n_facilities.py
- /test
- map.R
- .gitignore
- .pre_commit_config.yaml
- README.md
- requirements.txt
- requirements_dev.txt
```

## Create virtual environment (optional but suggest)

```plaintext
python3 -m venv <venv_name> python=3.10
source <venv_name>/bin/activate
```

## Used packages

```plaintext
pip3 install -r requirements.txt
```

## Function using

**Examples of functions calling inside `/src`**

```python
from .utils import (
    load_data, one_hot_encoding,
    add_twd97_coordinates_to_dataframe as add_twd97,
    add_wgs84_coordinates_to_dataframe as add_wgs84
)
```

**Examples of inheritance of classes inside `/src`**

```python
from .father_class import FatherClass

class YourClass(FatherClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

**Examples of `src/predict.py`**

```plaintext
python -m src.predict --dims <number of features>
```

argument `--dims` is the number of features in use.

## Execute

Plz execute on the root directory

```plaintext
python -m src.python_file
```
