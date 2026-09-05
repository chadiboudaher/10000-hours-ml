# Residual Connections and Network Degradation

This experiment investigates how residual connections help mitigate the degradation problem in deep neural networks.

## Experiment

Four CNNs were trained on CIFAR-10 under the same training configuration:

- Plain CNN — 4 blocks
- Plain CNN — 16 blocks
- Residual CNN — 4 blocks
- Residual CNN — 16 blocks

Each block contains two convolutional layers. The residual version adds a skip connection:

$$
y = F(x) + x
$$

All models were trained for 10 epochs using Adam with a learning rate of \(10^{-3}\).

## Results

| Model       | Train Accuracy | Test Accuracy |
| ----------- | -------------: | ------------: |
| Plain-4     |         85.44% |        77.64% |
| Plain-16    |         69.12% |        65.30% |
| Residual-4  |         83.32% |        73.69% |
| Residual-16 |         87.65% |        77.24% |

Increasing the plain network from 4 to 16 blocks reduced training accuracy from **85.44% to 69.12%**, demonstrating degradation despite the deeper model having greater capacity.

Adding residual connections to the 16-block network increased training accuracy from **69.12% to 87.65%** and test accuracy from **65.30% to 77.24%**.

## Conclusion

The experiment shows that simply increasing network depth can make optimization more difficult.

Residual connections allow a block to learn a residual transformation:

$$
F(x) = H(x) - x
$$

and, when an additional transformation is not useful, the residual can approach zero so that the block behaves approximately like an identity mapping.

The results support the idea that residual connections make deeper neural networks easier to optimize and help mitigate the degradation problem.
