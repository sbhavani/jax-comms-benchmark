import jax
import jax.numpy as jnp
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define a simple JAX function
def compute_sum(x):
    return jnp.sum(x)

# Create a random array on each process
x = jax.random.normal(jax.random.PRNGKey(rank), (10,))

# Compute the local sum
local_sum = compute_sum(x)

# Reduce the local sums to the root process
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum: {total_sum}")
