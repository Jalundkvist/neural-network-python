# Neural network from scratch in Python


### Developed by: Jacob Lundkvist & Jacob Nilsson
### Examiner: Erik Pihl

## Overview
The goal of this project was to construct a neural network from the ground up without utilizing any external libraries. To verify the neural network it was given the training data for a 4-to-1 XOR gate to later predict the output from four push buttons connected to a Raspberry Pi. According to the task, a LED should light up if an odd number of buttons are pressed simultaneously, and be turned off when an even number of buttons are pressed.

PDF of [assignment](docs/Projekt%20II%20%E2%80%93%20Neuralt%20n%C3%A4tverk%20i%20ett%20inbyggt%20system.pdf) (swedish)

## Results
The results obtained were promising, with no major obstacles encountered during the development process. The network met all requirements and performed without any errors. The training data was read from a file defined in [main](example/main.py), standard [train_data.txt](example/train_data.txt) and was trained based on the settings passed in the main function. The output of the network was written to the file [output.txt](example/output.txt). The optimal settings for the network were a network with three layers and with 4 nodes each with settings of 10,000 epochs and a learning rate of 0.0255, resulting in a high level of accuracy while still being able to train quickly. 

A limitation that still persists is the issue of random initialization of all weights and biases, which could potentially result in underfitting and a resulting accuracy below acceptable levels. This issue was addressed by providing the user with the ability to set a desired accuracy threshold and prompting them to re-run the code (or quit) if training fails to meet the specified threshold. This approach allows for greater control and flexibility in achieving the desired level of accuracy.

![alt text](https://github.com/Jalundkvist/neural-network-python/blob/main/docs/training_example.png?raw=true)
![alt text](https://github.com/Jalundkvist/neural-network-python/blob/main/docs/failed_training_example.png?raw=true)
## Discussion
The project was on a good difficulty and was an enjoyable experience to work on. It was an educational and enlightening experience to learn how to construct and apply neural networks. The guidance provided by Erik in the form of scaled down examples in both C and C++ was invaluable and was a great help in understanding any obstacles encountered during the development process.
