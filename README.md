# Byte_Segment_Neural_Network-for-Network-Traffic-Classification
## Introduce

Reproduction of the Byte Segment Neural Network for Network Traffic Classification

I have tried to reproduce this paper, but it may contain some errors and code irregularities. After I trained 5 epochs I got 95% accuracy. The model in the paper obtained higer accuracy after 20 epochs of training.

## Usage

```
python add_model.py
```

if you want to change the gamma and alpha of the loss function ,you need to change it in focal_loss.py.

you can change the config in the add_model.py[13-16]

## Paper

```latex
@INPROCEEDINGS{8624128,  author={R. {Li} and X. {Xiao} and S. {Ni} and H. {Zheng} and S. {Xia}},  booktitle={2018 IEEE/ACM 26th International Symposium on Quality of Service (IWQoS)},   title={Byte Segment Neural Network for Network Traffic Classification},   year={2018},  volume={},  number={},  pages={1-10},  doi={10.1109/IWQoS.2018.8624128}}
```

