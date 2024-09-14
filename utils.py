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

def print_header(comm, comm_op, world_size=2, bw_unit=DEFAULT_UNIT, raw=False):
    tput = f'Throughput ({bw_unit})'
    busbw = f'BusBW ({bw_unit})'
    header = f"\n---- Performance of {comm_op} on {world_size} devices ---------------------------------------------------------\n"
    duration_str = 'Duration'
    if raw:
        duration_str += ' (us)'
    header += f"{'Size (Bytes)':20s} {'Description':25s} {duration_str:20s} {tput:20s} {busbw:20s}\n"
    header += "----------------------------------------------------------------------------------------------------"
    print_rank_0(header)


def get_bw(comm, comm_op, size, avg_duration, bw_unit=DEFAULT_UNIT):
    n = get_world_size(comm)
    tput = 0
    busbw = 0
    if comm_op == "all_to_all":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt" or comm_op == "broadcast":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("wrong comm_op specified")
        exit(0)

    if bw_unit == 'Gbps':
        tput *= 8
        busbw *= 8

    return tput, busbw

def get_metric_strings(tput, busbw, duration, raw=False):
    duration_ms = duration * 1e3
    duration_us = duration * 1e6
    tput = f'{tput / 1e9:.3f}'
    busbw = f'{busbw /1e9:.3f}'

    if duration_us < 1e3 or raw:
        duration = f'{duration_us:.3f}'
        if not args.raw:
            duration += ' us'
    else:
        duration = f'{duration_ms:.3f} ms'
    return tput, busbw, duration


# Helper function to pretty-print message sizes
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def print_rank_0(comm, message):
    if get_rank(comm) == 0:
        print(message)
