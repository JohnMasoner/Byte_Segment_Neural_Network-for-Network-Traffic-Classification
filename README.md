# Byte_Segment_Neural_Network-for-Network-Traffic-Classification
## Introduce

Reproduction of the Byte Segment Neural Network for Network Traffic Classification

[Paper]: https://ieeexplore.ieee.org/document/8624128	"Byte Segment Neural Network for Network Traffic Classification"

I have tried to reproduce this paper, but it may contain some errors and code irregularities. After I trained 5 epochs I got 95% accuracy. The model in the paper obtained higer accuracy after 20 epochs of training.

## Usage

```
python add_model.py
```

if you want to change the gamma and alpha of the loss function ,you need to change it in focal_loss.py.

you can change the config in the add_model.py[13-16]