
import numpy as np
import matplotlib.pyplot as plt

# Define your custom function (e.g., sine wave, cosine wave, etc.)
def custom_function(t):
    return np.sin(t/10)  # Replace with any other function


# Leaky ReLU activation function
def leaky_relu(x):
    return np.maximum(0.1 * x, x)


# Layer class to represent individual layers of the RNN
class Layer:
    def __init__(self, input_size, output_size):
        # Xavier initialization for weights
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.random.randn(output_size)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        self.y = np.dot(x, self.weights) + self.bias
        return leaky_relu(self.y)

    def backward(self, grad):
        grad = grad * (self.y > 0) + grad * 0.1 * (self.y <= 0)

        # Gradient clipping
        max_grad_norm = 5.0  # Adjust this threshold if needed
        grad = np.clip(grad, -max_grad_norm, max_grad_norm)

        self.grad_weights = np.dot(self.x.T.reshape(-1, 1), grad.reshape(1, -1))
        self.grad_bias = grad
        grad_input = np.dot(grad, self.weights.T)
        return grad_input, self.grad_weights, self.grad_bias

    def update(self, lr):
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias


# RNN class to represent the recurrent neural network
class RNN:
    def __init__(self, layer_size):
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(Layer(layer_size[i], layer_size[i + 1]))

    def forward(self, x):
        # Process the sequence step by step through each layer
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad, grad_weights, grad_bias = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)


# Function to generate data based on an arbitrary function
def generate_function_data(func, seq_length=1000, time_steps=100):
    t = np.linspace(0, seq_length, seq_length)  # Time steps
    target_function = func(t)  # Apply the function to the time steps

    # Prepare input-output sequences
    x = []
    y = []
    for i in range(len(target_function) - time_steps):
        x.append(target_function[i:i + time_steps])  # Sequence of time_steps values
        y.append(target_function[i + time_steps])  # The next value (target)

    return np.array(x), np.array(y), target_function

def main():
    # Generate data based on the custom function
    time_steps = 10  # Length of the input sequence
    seq_length = 100  # Length of the function sequence
    x_data, y_data, target_function = generate_function_data(custom_function, seq_length, time_steps)

    # Visualize the target function (e.g., sine wave)
    plt.plot(np.linspace(0, seq_length, seq_length), target_function, label="Target Function")
    plt.title(f"Generated Target Function")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Create RNN model
    rnn = RNN([time_steps, 40, 30, 20, 10, 10, 1])

    # Training parameters
    learning_rate = 0.01  # Lower the learning rate for stable training
    epochs = 1000
    losses = []

    # Train the model
    for epoch in range(epochs):
        epoch_loss = 0

        # Loop through each sequence
        for i in range(len(x_data)):
            x = x_data[i]
            y = y_data[i]

            # Forward pass
            prediction = rnn.forward(x)

            # Calculate loss (Mean Squared Error)
            loss = np.square(prediction - y).mean()
            epoch_loss += loss

            # Backward pass
            grad = 2 * (prediction - y)  # Derivative of MSE
            rnn.backward(grad)

            # Update weights
            rnn.update(learning_rate)

        # Track the loss for every epoch
        losses.append(epoch_loss / len(x_data))

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(x_data)}")

        learning_rate *= 0.999  # Optional: Decrease the learning rate over epochs

    # Plot the training loss over epochs
    plt.plot(range(epoch+1), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Test the model
    test_input = x_data[0]  # Use the first sequence for testing
    predictions = []

    # Predict the next value using the trained RNN
    for _ in range(len(y_data)):
        pred = rnn.forward(test_input)
        predictions.append(pred)
        # Update the input for the next prediction
        test_input = np.roll(test_input, -1)  # Shift the sequence by 1
        test_input[-1] = pred  # Add the predicted value as the new input

    # Plot the results
    plt.plot(np.arange(len(y_data)), y_data, label="True Function", color='blue')
    plt.plot(np.arange(len(predictions)), predictions, label="Predicted Function", linestyle="dashed", color='red')
    plt.title(f"True vs Predicted Function")
    plt.legend()
    plt.show()

main()
