# Unsupervised-meta-review-generation

## Preprocessing

To run our framework, you need to preprocess your dataset and extract opinion phrases from the input sentences.


## Running

The workflow has following 4 steps.

- Step 1. Data preparation
- Step 2. Training
- Step 3. Aggregation
- Step 4. Generation

You can skip Steps 1-3 by downloading our pre-trained model and dataset.

### Step 1. Data preparation

```
python src/prepare_new.py \
  config/prepare_default.json
```

### Step 2. Training
```
$ pip install -r requirements.txt
$ cd meta_review
$ python src/train_new.py \
  config/prepare_default.json \
  config/train_default.json
```
