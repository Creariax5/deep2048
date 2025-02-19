from sklearn.datasets import load_digits
import sk_dataset

def get_digits():
    digits = load_digits()
    X = digits.data
    y = digits.target
    y = sk_dataset.convert_to_one_hot(y, 10)
    print("X: ", X.shape)
    print("y: ", y.shape)
    return X, y

def get_2inp_3out_dataset():
    X, y = sk_dataset.get_dataset()
    y = sk_dataset.convert_to_one_hot(y, 3)
    print("X: ", X.shape)
    print("y: ", y.shape)
    return X, y