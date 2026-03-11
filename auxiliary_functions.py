import numpy as np
import random as rd
import networkx as nx
from scipy.fft import fft, ifft, fftfreq

def derivative(v1, v2):
    """
    Compute the numerical derivative of v1 with respect to v2 using a central difference method.

    Parameters:
    v1 (list or numpy array): Array of function values.
    v2 (list or numpy array): Array of independent variable values.

    Returns:
    list: Numerical derivative of v1 with respect to v2.
    """
    assert len(v1) == len(v2), "Error: in order to compute the numerical derivative, the two variables need to have the same length."
    
    derivative = []
    for i, vi in enumerate(v1[1:]):
        m1 = (vi - v1[i-1]) / (v2[i] - v2[i-1])
        m2 = (v1[i+1] - vi) / (v2[i+1] - v2[i])
        derivative.append((m1 + m2) / 2)
    
    return derivative

def numerical_integral(x, f):
    """
    Compute the numerical integral of f with respect to x using the trapezoidal rule.
    
    Parameters:
    x (numpy array): Array of x values.
    f (numpy array): Array of f(x) values.
    
    Returns:
    float: Numerical integral of f with respect to x.
    """
    return np.trapz(f, x)


def segmented_integrals(x, f, x_splits):
    """
    Compute integrals of f with respect to x over subintervals defined by x_splits.
    
    Parameters:
    x (numpy array): Array of x values.
    f (numpy array): Array of f(x) values.
    x_splits (list of int): List of indices defining the subintervals.
    
    Returns:
    list of float: List of integrals for each subinterval.
    """
    integrals = []
    # Include 0 and len(x) to cover the first and last segments
    splits = [0] + x_splits + [len(x)]
    
    for i in range(len(splits) - 1):
        start_idx = splits[i]
        end_idx = splits[i + 1]
        integral = numerical_integral(x[start_idx:end_idx], f[start_idx:end_idx])
        integrals.append(integral)
    
    return integrals


def running_mean(x, N):
    """
    Compute the moving average of the input array x with a window size of N.

    Parameters:
    x (list or numpy array): Array of values to compute the moving average for.
    N (int): Window size for the moving average.

    Returns:
    numpy array: Array of moving averages.
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def percentile_mask(v, q):
    """
    Generates a mask vector with ones at the positions of v that are in the n-th percentile and 0 otherwise.

    Parameters:
    v (np.ndarray): Input array.
    q (float): Percentile (0-100).

    Returns:
    np.ndarray: Mask vector with ones at the positions of v that are in the n-th percentile and 0 otherwise.
    """
    # Calculate the percentile value
    percentile_value = np.percentile(v, q)
    
    # Generate the mask vector
    mask = (v >= percentile_value).astype(int)
    
    return mask


def generate_random_colors(n):
    '''
    Generates a list of `n` random colors in hexadecimal format.

    The function creates `n` unique colors by generating random 
    hexadecimal color codes. These color codes are formatted as 
    strings in the format "#RRGGBB", where RR, GG, and BB are 
    two-digit hexadecimal numbers representing the red, green, 
    and blue color channels, respectively.

    Parameters:
    -----------
    n : int
        The number of random colors to generate.

    Returns:
    --------
    colors : list of str
        A list of `n` strings, each representing a random color 
        in hexadecimal format.

    Example:
    --------
    >>> generate_random_colors(5)
    ['#1a2b3c', '#4d5e6f', '#7a8b9c', '#c1d2e3', '#e4f5a6']
    
    Notes:
    ------
    The colors are generated randomly, so each call to this 
    function will produce a different set of colors.
    '''
    colors = []
    for i in range(n):
        color = "#%06x" % rd.randint(0, 0xFFFFFF)
        colors.append(color)
    return colors


def stochastic_block_model(num_blocks, block_sizes, p_within_block, q_between_blocks):
    '''
    Generates a stochastic block model (SBM) graph.

    Parameters:
    -----------
    num_blocks : int
        The number of blocks (or communities) in the graph.

    block_sizes : int or list of int
        If an integer is provided, it indicates that each block should have the same size.
        If a list is provided, it must have a length equal to num_blocks, where each element represents 
        the size of the corresponding block. 

    p_within_block : float
        The probability of edges within the same block.

    q_between_blocks : float
        The probability of edges between different blocks.

    Returns:
    --------
    graph : networkx.Graph
        A graph object representing the stochastic block model.
    
    Raises:
    -------
    ValueError:
        If block_sizes is not a list of length equal to num_blocks and is not an integer.
    '''

    # Check if block_sizes is an integer or a list
    if isinstance(block_sizes, int):
        # Create a list with equal block sizes
        block_sizes = [block_sizes] * num_blocks
    elif isinstance(block_sizes, list):
        # Check if the length of block_sizes matches num_blocks
        if len(block_sizes) != num_blocks:
            raise ValueError("The length of block_sizes must be equal to num_blocks.")
    else:
        # Raise an error if block_sizes is neither an int nor a list
        raise ValueError("block_sizes must be either an integer or a list of length equal to num_blocks.")

    # Generate block probabilities matrix
    prob_matrix = [[p_within_block if i == j else q_between_blocks for j in range(num_blocks)] for i in range(num_blocks)]

    # Generate stochastic block model graph
    graph = nx.stochastic_block_model(block_sizes, prob_matrix)

    return graph

def Discrete_RW_T(A):    # Calculate the degree vector (sum of non-zero elements in each row)
    degree_vector = np.array(A.sum(axis=1)).ravel()

    # Avoid division by zero by setting degree to 1 for nodes with no edges
    no_out_edges = degree_vector == 0
    degree_vector[no_out_edges] = 1.0

    # Create a diagonal matrix from the inverted degree vector
    degree_matrix = np.diag(1.0 / degree_vector)

    # Calculate the random walk Transition matrix: T = D^(-1) * A
    T = degree_matrix @ A
    np.fill_diagonal(T, no_out_edges, wrap=False)
    return T