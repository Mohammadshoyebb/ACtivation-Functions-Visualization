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
   
