# Confederated Learning: Going Beyond Centralization

This is a Pytorch implementation of the methods described in our paper:

>Z. Wang, Q. Xu, K. Ma, X. Cao and Q. Huang. Confederated Learning: Going Beyond Centralization. MM2022.

## Dependencies
- Pytorch >= 1.9.0
- numpy

## Data
We perform evaluations on the following benchmark datasets: (a)`MNIST` $\to$ `USPS`; (b) the Office-31 dataset; (c) `Office` $\to$ `Caltech`. For each dataset, we create a corresponding folder in the root folder, and the `dataset.py` describes how the data is organized. 

For `MNIST` and `USPS`, you can get the data via Pytorch.
For the Office-31 dataset and `Office` $\to$ `Caltech`, all the data can be found in their [homepage](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code).

## Train and test

For each dataset, the source model is available via
```bash
python train_source.py
```
and its performance on the target dataset is available via
```bash
python test_source_on_target.py
```
Besides, you can train the baseline model and the proposed methods via
```bash
python train_target_raw.py
python train_target_reweighting.py
python train_target_ensemble.py
python train_target_reg.py
```

## Citation

```
@inproceedings{DBLP:conf/mm/WangX0CH21,
  author    = {Zitai Wang and
               Qianqian Xu and
               Ke Ma and
               Xiaochun Cao and
               Qingming Huang},
  title     = {Confederated Learning: Going Beyond Centralization},
  booktitle = {{ACM} Multimedia Conference},
  year      = {2022},
}
```