import numpy as np

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