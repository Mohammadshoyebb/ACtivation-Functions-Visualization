### Activation Functions Visualization

This repository demonstrates various activation functions commonly used in neural networks. Each function is visualized using matplotlib to understand its behavior across a range of input values.

---

#### Overview

The code provided visualizes the following activation functions:

1. **Binary Step**: Outputs 1 if the input is greater than 0, otherwise 0.
   
2. **Piecewise Linear**: A linear function with different slopes in different intervals.
   
3. **Bipolar**: Outputs 1 if the input is greater than 0, otherwise -1.
   
4. **Sigmoid**: S-shaped curve that squashes input values between 0 and 1.
   
5. **Bipolar Sigmoid**: Similar to sigmoid but outputs values between -1 and 1.
   
6. **Hyperbolic Tangent (TanH)**: S-shaped curve that outputs values between -1 and 1.
   
7. **ArcTangent**: Outputs the inverse tangent (arctan) of the input values.
   
8. **Rectified Linear Unit (ReLU)**: Outputs 0 if input is less than 0, otherwise outputs the input itself.
   
9. **Leaky Rectified Linear Units (Leaky ReLU)**: Variant of ReLU with a small non-zero gradient for negative inputs.
   
10. **Exponential Linear Units (ELU)**: Smooth approximation of ReLU for negative inputs.
   
11. **SoftPlus**: Smooth approximation of ReLU with asymptotic behavior.
   
12. **Gaussian (or Gaussian Error Linear Unit, GELU)**: Activation function based on the Gaussian distribution.
   
13. **Swish Activation**: Recently proposed activation function that performs well in deep neural networks.
   
14. **Mish Activation**: Novel activation function showing promising results in various architectures.
   
15. **Softsign Activation**: Scaled version of the input value.
   
16. **Softmax Function**: Converts logits (raw predictions) into probabilities.

---

#### Implementation Details

The provided code uses NumPy and Matplotlib to plot each activation function's behavior over a specified range of input values. Each plot showcases the function's output characteristics, aiding in understanding their applications in neural networks.

---

#### Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt

# Ensure plots appear in the Jupyter Notebook
%matplotlib inline

# Generate input values
x = np.arange(-5, 5, 0.01)

def plot(func, yaxis=(-1.4, 1.4), title=''):
    """
    Plot the given activation function over the range of input values.
    
    Parameters:
    - func: Activation function to be plotted.
    - yaxis: Tuple specifying the y-axis limits for the plot.
    - title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    plt.ylim(yaxis)
    plt.locator_params(nbins=5)
    plt.xticks(fontsize=14)
    plt.axhline(lw=1, c="black")
    plt.axvline(lw=1, c="black")
    plt.grid(alpha=0.4, ls='-.')
    plt.box(on=None)
    plt.plot(x, func(x), c="r", lw=3)
    plt.title(title)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()

# Define each activation function with a lambda function or regular function
# and plot using the 'plot' function defined above.

# 1. Binary Step
binary_step = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[float])
plot(binary_step, yaxis=(-0.4, 1.4), title='Binary Step Activation')

# 2. Piecewise Linear
piecewise_linear = np.vectorize(lambda x: 1 if x > 3 else 0 if x < -3 else 1/6 * x + 1/2, otypes=[float])
plot(piecewise_linear, title='Piecewise Linear Activation')

# 3. Bipolar
bipolar = np.vectorize(lambda x: 1 if x > 0 else -1, otypes=[float])
plot(bipolar, yaxis=(-1.4, 1.4), title='Bipolar Activation')

# 4. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
plot(sigmoid, yaxis=(-0.4, 1.4), title='Sigmoid Activation')

# 5. Bipolar Sigmoid
def bipolar_sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))
plot(bipolar_sigmoid, yaxis=(-1.4, 1.4), title='Bipolar Sigmoid Activation')

# 6. Hyperbolic Tangent (TanH)
def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1
plot(tanh, yaxis=(-1.4, 1.4), title='TanH Activation')

# 7. ArcTangent
def arctan(x):
    return np.arctan(x)
plot(arctan, yaxis=(-1.4, 1.4), title='ArcTangent Activation')

# 8. Rectified Linear Unit (ReLU)
relu = np.vectorize(lambda x: x if x > 0 else 0, otypes=[float])
plot(relu, yaxis=(-0.4, 1.4), title='ReLU Activation')

# 9. Leaky Rectified Linear Units (Leaky ReLU)
leaky_relu = np.vectorize(lambda x: max(0.1 * x, x), otypes=[float])
plot(leaky_relu, yaxis=(-0.4, 1.4), title='Leaky ReLU Activation')

# 10. Exponential Linear Units (ELU)
elu = np.vectorize(lambda x: x if x > 0 else 0.5 * (np.exp(x) - 1), otypes=[float])
plot(elu, yaxis=(-0.4, 1.4), title='ELU Activation')

# 11. SoftPlus
def softplus(x):
    return np.log(1 + np.exp(x))
plot(softplus, yaxis=(-0.4, 1.4), title='SoftPlus Activation')

# 12. Gaussian (GELU)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
plot(gelu, yaxis=(-0.4, 1.4), title='GELU Activation')

# 13. Swish Activation
def swish(x):
    return x / (1 + np.exp(-x))
plot(swish, yaxis=(-0.4, 1.4), title='Swish Activation')

# 14. Mish Activation
def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))
plot(mish, yaxis=(-0.4, 1.4), title='Mish Activation')

# 15. Softsign Activation
def softsign(x):
    return x / (1 + np.abs(x))
plot(softsign, yaxis=(-0.4, 1.4), title='Softsign Activation')

# 16. Softmax Function (Plotting example)
def softmax(logits):
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits)
    return softmax_probs

logits = np.array([2.0, 1.0, 0.1])
softmax_probs = softmax(logits)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(len(logits)), logits, color='blue')
plt.title('Logits')
plt.xlabel('Class')
plt.ylabel('Logit Value')
plt.xticks(range(len(logits)))

plt.subplot(1, 2, 2)
plt.bar(range(len(softmax_probs)), softmax_probs, color='green')
plt.title('Softmax Probabilities')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.xticks(range(len(softmax_probs)))
plt.ylim([0, 1])

plt.tight_layout()
plt.show()
