from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
import jax.lax as lax
import time

from constants import *
from utils import *

def all_reduce(x, comm):
    num_local_devices = jax.local_device_count()
    # Reshape x to have a leading axis size equal to the number of local devices
    x = x.reshape((num_local_devices,) + x.shape)
    # Perform the allreduce operation
    return comm.allreduce(x, op=MPI.SUM)


def run_all_reduce(comm, dtype, maxsize, mem_factor, scan=True):
    world_size = get_world_size(comm)
    global_rank = get_rank(comm)
    local_rank = get_rank(comm)

    # Prepare benchmark header
    print_header(comm, 'all_reduce', world_size)

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
                mat = jnp.ones((world_size, M), dtype=dtype)
                #sync_all()
                x = ((mat * float(global_rank)).reshape(-1))
                del mat
                jax.device_put(x, jax.devices()[local_rank])
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if get_rank(comm) == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    break
                else:
                    raise e
            timed_all_reduce(x, comm, start_time, end_time)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so we double,mem_factor
        elements_per_gpu = max_numel(dtype=dtype,
                                     mem_factor=mem_factor * 2,
                                     local_rank=local_rank)

        try:
            mat = jnp.ones(elements_per_gpu, dtype=dtype)
            x = ((mat * float(global_rank)).reshape(-1))
            jax.device_put(x, jax.devices()[local_rank])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if get_rank(comm) == 0:
                    print('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                return
            else:
                raise e

        timed_all_reduce(x, comm, start_time, end_time)

def timed_all_reduce(x, comm, start_time, end_time, warmups=10, trials=10):
    for i in range(warmups):
        all_reduce(x, comm)

    if start_time is None:
        start_time = time.time()
    for j in range(trials):
        all_reduce(x, comm)
    if end_time is None:
        end_time = time.time()
    duration = end_time - start_time
    #print(f"All-reduce time: {duration:.6f} seconds")

    # maintain and clean performance data
    avg_duration = duration / trials
    # JAX arrays don't ahve element_size()
    x_np = np.array(x)
    element_size = x.dtype.itemsize
    size = element_size * x.size
    n = get_world_size(comm)
    tput, busbw = get_bw(comm, 'all_reduce', size, avg_duration)
    tput_str, busbw_str, duration_str = get_metric_strings(tput, busbw, avg_duration)
    desc = f'{x.size}x{x.size}'

    size = convert_size(size)
    #print(f"{size:<20} {desc:25s} {avg_duration:20}")
    print_rank_0(comm, f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")

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
