import numpy as np
import matplotlib.pyplot as plt

# Load in the training data .csv file
train_data = np.loadtxt('data/training_set.csv', delimiter=',')
train_labels = np.loadtxt('data/training_labels_bin.csv', delimiter=',')

# Concatenate the labels to the training data
train_data = np.concatenate((train_data, train_labels), axis=1)

# Do the same thing for validation data
val_data = np.loadtxt('data/validation_set.csv', delimiter=',')
val_labels = np.loadtxt('data/validation_labels_bin.csv', delimiter=',')
val_data = np.concatenate((val_data, val_labels), axis=1)

# Save the data to a .csv file
np.savetxt('data/train_data.csv', train_data, delimiter=',')
np.savetxt('data/val_data.csv', val_data, delimiter=',')

def sig(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function for the input x.
    """
    return 1 / (1 + np.exp(-x))

def sig_grad(x: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the sigmoid function for the input x.
    """
    return sig(x) * (1 - sig(x))

def loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the loss between the predicted and true labels.

    Args:
        y: The true labels.
        y_hat: The predicted labels.

    Returns:
        The loss between the predicted and true labels.
    """
    loss_val = 0.5 * np.sum((y - y_hat) ** 2)
    loss_grad = y_hat - y
    return loss_val, loss_grad

class DataLoader:
    """
    A custom data loader class for loading data from a .csv file.
    """
    def __init__(self, filename: str, shuffle: bool=True, normalize: bool=True) -> None:
        # Load data from file
        self.data = np.genfromtxt(filename, delimiter=',')
        self.features = self.data[:, :-3]
        self.labels = self.data[:, -3:]

        self.batch_size = 1 # batch size of 1 as per constraints
        self.shuffle = shuffle
        self.indexes = np.arange(self.features.shape[0])

        # Normalize data if required
        if normalize:
            self.features = self.normalize(self.features)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        # Standardize data: mean of 0 and standard deviation of 1
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / std
    
    def __len__(self) -> int:
        # Return the number of batches
        return int(np.ceil(len(self.features) / self.batch_size))

    def __iter__(self) -> 'DataLoader':
        self.indexes = np.arange(self.features.shape[0])
        # Shuffle data at the start of each iteration if required
        if self.shuffle:
            np.random.shuffle(self.indexes)
        return self

    def __next__(self) -> tuple:
        # If there's no more data to process, stop iteration
        if len(self.indexes) == 0:
            raise StopIteration

        # Select indexes for the next batch
        batch_indexes = self.indexes[:self.batch_size]
        self.indexes = self.indexes[self.batch_size:]

        # Return the batch of data
        return self.features[batch_indexes], self.labels[batch_indexes]
    
class MLP:
    def __init__(self, layer_sizes: list, lr: float=0.01) -> None:
        self.lr = lr

        # Initialize weights with random values. Note that the output layer does not have weights.
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1 / layer_sizes[i]))

        # Initialize bias terms with random values. Note that the output layer does not have bias terms.
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.biases.append(np.random.rand(layer_sizes[i+1]))
        
        # Initialize hidden states with zeros. Note that the first hidden state is the input, and the last hidden state is the output layer.
        self.hidden_states = []
        for i in range(len(layer_sizes)):
            self.hidden_states.append(np.zeros(layer_sizes[i]))

    def _forward(self, x: np.ndarray) -> list:
        self.hidden_states[0] = x
        for i in range(len(self.weights)):
            self.hidden_states[i+1] = np.dot(x, self.weights[i]) + self.biases[i]
            x = sig(self.hidden_states[i+1])

        output = sig(self.hidden_states[-1])

        return output
    
    def _backward(self, loss_grad: float) -> tuple:
        # Make an array of the same size as the hidden states to store the gradients
        self.pre_gradients = [np.zeros_like(h_state) for h_state in self.hidden_states]
        self.post_gradients = [np.zeros_like(h_state) for h_state in self.hidden_states]

        # Get the gradient of the loss with respect to the output layer
        self.post_gradients[-1] = loss_grad

        for i in range(len(self.hidden_states) - 1, 0, -1):
            # Compute the pre-activation gradient for the current layer
            self.pre_gradients[i] = self.post_gradients[i] * sig_grad(self.hidden_states[i])

            # Update the weights and biases for the current layer
            self.weights[i-1] -= self.lr * np.dot(self.hidden_states[i-1].T, self.pre_gradients[i])
            self.biases[i-1] -= self.lr * self.pre_gradients[i][0]

            # Compute the gradient of the loss with respect to the hidden states
            self.post_gradients[i-1] = np.dot(self.pre_gradients[i], self.weights[i-1].T)
    
    def _loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        loss_val = 0.5 * np.sum((y - y_hat) ** 2)
        loss_grad = y_hat - y
        return loss_val, loss_grad

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int=10) -> tuple:
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in train_loader:
                # Forward pass
                y_hat = self._forward(x)[-1]

                # Compute loss
                loss_val, loss_grad = self._loss(y, y_hat)
                epoch_loss += loss_val

                # Backward pass
                self._backward(loss_grad)

            # Compute average loss for the epoch
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)

            # Compute validation loss
            val_loss = 0
            for x, y in val_loader:
                y_hat = self._forward(x)[-1]
                val_loss += self._loss(y, y_hat)[0]
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f'Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}')

            if epoch > 10 and epoch % 10 == 0:
                self.lr /= 3

        return train_losses, val_losses
    
def main () -> None:
    # Load data
    train_loader = DataLoader('data/train_data.csv')
    val_loader = DataLoader('data/val_data.csv')

    # Initialize MLP
    mlp = MLP([354, 50, 50, 3])

    # Train MLP
    train_losses, val_losses = mlp.train(train_loader, val_loader, epochs=100)

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()