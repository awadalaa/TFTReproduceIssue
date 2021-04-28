# TFT ReproduceIssue

This repository is to reproduce an issue with a Tensorflow and Tensorflow-Transform pipeline. The issue is here https://github.com/tensorflow/transform/issues/237

## QuickStart

```
python -m venv ~/venv
source ~/venv/bin/activate

pip install -r requirements.txt

python -m data.task

python -m trainer.task

python -m inference.task
```
