# Adaline Neural Network in Python

This code is an implementation of a Adaline Neural Network in Python. It generates synaptic weights based on a training file. Once the synaptic weights have been found based on Delta rule, we can run tests on another tab of the worksheet and see the predicted answers. All the logs of the execution are saved in the `output.txt` file.

## Dependencies

- pandas

## Usage

- The input data should be provided in an excel file with separate sheets for training and testing.
- The training and testing data should have the same columns, with the output variable in the last column.
- Set the learning rate, interaction amount limit and precision as needed.
- Run the script with `python main.py`.
