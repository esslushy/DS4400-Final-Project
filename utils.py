import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def gather_data(to_drop: list[str]=[]) -> tuple:
    """
      Automatically gets our data from the World Happiness Report file and drops the nans.

      Args:
        to_drop: Any additional columns to drop from the dataframe.

      Returns:
        The train data, the test data, the train labels, and the test labels in numpy arrays.
    """
    train_df, test_df = pd.read_csv('data/train.csv'), pd.read_csv("data/test.csv")
    train_labels, test_labels = train_df.pop("Life Ladder"), test_df.pop("Life Ladder")
    # Purge unused columns
    to_drop.extend(["Country name", "Year"])
    for col in to_drop:
        train_df.pop(col)
        test_df.pop(col)
    # Min-max scaling
    gdp_scaling = MinMaxScaler()
    life_scaling = MinMaxScaler()
    train_df["Log GDP per capita"] = gdp_scaling.fit_transform(train_df["Log GDP per capita"].values.reshape(-1, 1))
    train_df["Healthy life expectancy at birth"] = life_scaling.fit_transform(train_df["Healthy life expectancy at birth"].values.reshape(-1, 1))
    test_df["Log GDP per capita"] = gdp_scaling.transform(test_df["Log GDP per capita"].values.reshape(-1, 1))
    test_df["Healthy life expectancy at birth"] = life_scaling.transform(test_df["Healthy life expectancy at birth"].values.reshape(-1, 1))
    print(train_df.columns)
    return train_df.to_numpy(), test_df.to_numpy(), train_labels.to_numpy(), test_labels.to_numpy()

def basis_expansion(x: np.ndarray, n: int, bias: bool=True) -> np.ndarray:
    """
      Expands the degrees of freedom of the provided data by 
      degrees [1, 2, ..., n]. If bias is true, places a 1 at the end of data.

      Args:
        x: The data array in form of Nxd
        n: The number of degrees of freedom to expand to. Must be a positive number
        bias: Whether or not to include a bias value

      Returns:
        A numpy array which contains the expanded data.

      Raises:
        An exception if n < 1.
    """
    if(n < 1):
        raise Exception("n must be a positive integer.")    
    x_new = x
    for degree in range(2, n+1):
        x_new = np.concatenate([x_new, x**degree], axis=1)
    if bias:
        x_new = np.concatenate([x_new, np.ones((len(x), 1))], axis=1)
    return x_new

def mean_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
      Predicts the MSPE between the y_true and y_pred labels.

      Args:
        y_true: The true labels
        y_pred: the predicted labels

      Returns:
        An [0, 100] which represents the MSPE
    """
    return 100 * (1.0/len(y_true)) * np.sum(((y_true - y_pred)/y_true)**2)

# Testing
if __name__ == "__main__":
    x = np.array(
        [
          [4, 5, 6],
          [6, 7, 8]
        ]
    )
    print(basis_expansion(x, 3, True))
    print(basis_expansion(x, 1, False))
    train_data, train_label, test_data, test_label = gather_data()
    print(train_data[0])
    print(train_label[:3])
    print(test_data[0])
    print(test_label[:3])
    print(mean_squared_percentage_error(np.array([4, 5, 6, 7]), np.array([4.5, 4.5, 6, 8])))