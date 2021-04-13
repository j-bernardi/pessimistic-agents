# Distributed Q Learning - QuEUE

## Example

```bash

source set_path.sh  # set path to current dir

python main.py -h

python main.py --quantile 4 --mentor random_safe --trans 1 --num-episodes 100 --render 1
```
## Run unittests

```bash
python -m unittest discover tests
```

