import math
from constants import *

def get_world_size(comm):
    """
    Get the number of processes in the communicator.
    Args:
        comm (jax.experimental.mpi.Communicator): The communication object for MPI operations.
    Returns:
        int: The number of processes.
    """
    return comm.size

def get_rank(comm):
    """
    Get the rank of the current process in the communicator.
    Args:
        comm (jax.experimental.mpi.Communicator): The communication object for MPI operations.
    Returns:
        int: The rank of the current process.
    """
    return comm.rank

def print_header():
    return

def get_bw():
    return

def get_metric_strings():
    return

def convert_size():
    return

def print_rank_0(comm, message):
    if get_rank(comm) == 0:
        print(message)
