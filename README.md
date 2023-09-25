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
    - add_coordinates.py
- /test
- map.R
- .gitignore
- .pre_commit_config.yaml
- README.md
- requirements.txt
- requirements_dev.txt
```

## Used packages

```plaintext
pip3 install -r requirements.txt
```

## Function using

**Examples of functions calling inside `/src`**

```python
from .utils import (
    load_data, one_hot_encoding
)
from .add_coordinates import (
    add_twd97_coordinates_to_dataframe,
    add_wgs84_coordinates_to_dataframe
)
```

## Execute

Plz execute on the root directory

```plaintext
python -m src.python_file
```
