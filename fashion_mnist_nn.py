import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data= pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv").values
Y_train= data[:, 0]
X_train= data[:, 1:].T/255.0

test_data= pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv").values
Y_test= test_data[:, 0]
X_test= test_data[:, 1:].T/255.0
def ReLU(Z):
    return np.maximum(Z,0)

def softMax(Z):
    Z_shift= Z-np.max(Z, axis=0, keepdims=True)
    A= np.exp(Z_shift)
    return A/np.sum(A, axis=0, keepdims= True)

def ReLU_deriv(Z):
    return Z>0

def init_params(input_size=784, hidden_size= 128, output_size=10):
    W1= np.random.rand(hidden_size, input_size)* np.sqrt(2/input_size)
    b1= np.zeros((hidden_size, 1))
    W2= np.random.rand(output_size, hidden_size)* np.sqrt(2/hidden_size)
    b2= np.zeros((output_size, 1))
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1= W1.dot(X)+b1
    A1= ReLU(Z1)
    Z2= W2.dot(A1)+b2
    A2= softMax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_y= np.zeros((Y.max()+1, Y.size))
    one_hot_y[Y, np.arange(Y.size)]=1
    return one_hot_y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m= X.shape[1]
    one_hot_Y= one_hot(Y)
    dZ2= A2-one_hot_Y
    dW2=(1/m) *dZ2.dot(A1.T)
    db2= (1/m)* np.sum(dZ2, axis=1, keepdims= True)
    dZ1= W2.T.dot(dZ2)* ReLU_deriv(Z1)
    dW1= (1/m)*dZ1.dot(X.T)
    db1= (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1= W1-alpha*dW1
    b1= b1- alpha*db1
    W2= W2-alpha*dW2
    b2= b2- alpha*db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions==Y)/Y.size
    
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def gradient_descent(X, Y, alpha, iterations, hidden_size= 128):
    W1, b1, W2, b2= init_params(X.shape[0],hidden_size, 10)
    print(f"--- Training Neural Network ({X.shape[0]} inputs, {hidden_size} hidden units) ---")
    for i in range(iterations):
        Z1, A1, Z2, A2= forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2= back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2= update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i%(iterations//10)==0 or i==iterations-1:
            predictions=get_predictions(A2)
            accuracy=get_accuracy(predictions, Y)
            print(f"Iteration: {i:5d} | Accuracy: {accuracy:.4f}")
    return W1, b1, W2, b2

def test_prediction(index, W1, b1, W2,b2, X_data, Y_labels, class_names):
    current_image= X_data[:, index, None]
    prediction= make_predictions(current_image, W1, b1, W2, b2)[0]
    label=Y_labels[index]
    predicted_class= class_names[prediction]
    true_class= class_names[label]
    print(f"\n--- Prediction for Sample Index {index} ---")
    print(f"Prediction: {predicted_class} (Label {prediction})")
    print(f"True Label: {true_class} (Label {label})")
    current_image= current_image.reshape((28,28))*255
    plt.imshow(current_image, cmap='gray', interpolation='nearest')
    plt.title(f"Predicted: {predicted_class} | True: {true_class}")
    plt.show()
def run_fashion_nn():
    class_names=[
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    learningRate=0.1
    iterations=1000
    hiddenNodes=128
    W1, b1, W2, b2= gradient_descent(X_train, Y_train, learningRate, iterations, hiddenNodes)
    _,_,_,A2_test= forward_prop(W1, b1, W2, b2, X_test)
    test_predictions= get_predictions(A2_test)
    test_accuracy=get_accuracy(test_predictions, Y_test)
    print("\n--- Model Evaluation (Simulated Data) ---")
    print(f"Final Trained Model (Last Training Accuracy): {get_accuracy(get_predictions(A2_test), Y_test):.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("\n--- Visualizing Test Predictions ---")
    
    sample_indices = np.random.choice(X_test.shape[1], 5, replace=False)

    for index in sample_indices:
        test_prediction(index, W1, b1, W2, b2, X_test, Y_test, class_names)

if __name__ == '__main__':
    run_fashion_nn()
