import numpy as np
import pandas as pd

def gather_data(to_drop=[]) -> tuple:
    """
      Automatically gets our data from the World Happiness Report file and drops the nans.

      Args:
        to_drop: Any additional columns to drop from the dataframe.

      Returns:
        An Nxd numpy array for the data section, and a Nx1 numpy array for the labels
    """
    df = pd.read_csv('World Happiness Report 2005-2021.csv')
    df.dropna(inplace=True)
    label = df.pop("Life Ladder")
    # Purge unused columns
    to_drop.extend(["Country name", "Year"])
    [df.pop(col) for col in to_drop]
    return df.to_numpy(), label.to_numpy()

def basis_expanstion(x: np.ndarray, n: int, bias: bool=True) -> np.ndarray:
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

# Testing
if __name__ == "__main__":
    x = np.array(
        [
          [4, 5, 6],
          [6, 7, 8]
        ]
    )
    print(basis_expanstion(x, 3, True))
    print(basis_expanstion(x, 1, False))
    data, label = gather_data()
    print(data)
    print(label)