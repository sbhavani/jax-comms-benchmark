from mpi4py import MPI
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.mpi as jmpi

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

def all_reduce(x, comm):
    """
    Perform an all-reduce operation on the input tensor `x` using the provided communication object `comm`.

    Args:
        x (jax.numpy.ndarray): The input tensor to be reduced.
        comm (jax.experimental.mpi.Communicator): The communication object for MPI operations.

    Returns:
        jax.numpy.ndarray: The result of the all-reduce operation.
    """
    # Perform the all-reduce operation
    return comm.allreduce(x, op=lax.psum)


def run_all_reduce(comm, dtype, maxsize, mem_factor, scan=True):
    world_size = get_world_size(comm)
    global_rank = get_rank(comm)

    # JAX does not have CUDA events, so we will use a simple timer for timing
    start_time = None
    end_time = None

    if scan:
        M_LIST = []
        for x in (2**p for p in range(1, maxsize)):
            M_LIST.append(x)

        #sync_all() - consider comm.barrier()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = get_rank(comm)
            try:
                mat = jnp.ones((world_size, M), dtype=getattr(jnp, dtype))
                #sync_all()
                x = ((mat * float(global_rank)).reshape(-1))
                del mat
                jax.device_put(x, jax.devices()[local_rank])
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if get_rank(comm) == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    #sync_all()
                    break
                else:
                    raise e
            #sync_all()
            timed_all_reduce(x, comms, start_time, end_time)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so we double mem_factor
        elements_per_gpu = max_numel(dtype=getattr(jnp, dtype),
                                     mem_factor=mem_factor * 2,
                                     local_rank=local_rank)

        try:
            mat = jnp.ones(elements_per_gpu, dtype=getattr(jnp, dtype))
            x = ((mat * float(global_rank)).reshape(-1))
            jax.device_put(x, jax.devices()[local_rank])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if get_rank(comm) == 0:
                    print('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                #sync_all()
                return
            else:
                raise e
        #sync_all()
        timed_all_reduce(x, comms, start_time, end_time)

def timed_all_reduce(x, comms, start_time, end_time):
    if start_time is None:
        start_time = jax.default_backend().timer()
    all_reduce(x, comms)
    if end_time is None:
        end_time = jax.default_backend().timer()
    elapsed_time = end_time - start_time
    print(f"All-reduce time: {elapsed_time:.6f} seconds")


def max_numel(dtype, mem_factor=0.8, local_rank=0):
    """
    Calculate the maximum number of elements that can fit in GPU memory.

    Args:
        dtype (jax.numpy.dtype): The data type of the tensor.
        mem_factor (float): The fraction of GPU memory to use (default is 0.8).
        local_rank (int): The local rank of the GPU device (default is 0).

    Returns:
        int: The maximum number of elements that can fit in GPU memory.
    """
    # Get the device corresponding to the local rank
    device = jax.devices()[local_rank]

    # Get the total memory available on the device
    total_memory = device.memory_limit

    # Calculate the size of the data type in bytes
    dtype_size = jnp.dtype(dtype).itemsize

    # Calculate the maximum number of elements that can fit in the specified fraction of memory
    max_elements = int(total_memory * mem_factor / dtype_size)

    return max_elements

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD

    jax.distributed.initialize()
    dtype = jnp.bfloat16

    run_all_reduce(comm, dtype, maxsize=DEFAULT_MAXSIZE, mem_factor=0.8)

    # Finalize MPI
    MPI.Finalize()
