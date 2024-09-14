# jax-comms-benchmark
Benchmarking collective communication operations in JAX

## Example
`mpirun -np 4 python all_reduce.py`

---- Performance of all_reduce on 4 devices ---------------------------------------------------------
| Size (Bytes) | Description | Duration | Throughput (Gbps) | BusBW (Gbps) |
|--------------|-------------|----------|-------------------|--------------|
| 16.0 B | 8x8 | 837.922 us | 0.000 | 0.000 |
| 32.0 B | 16x16 | 911.617 us | 0.001 | 0.000 |
| 64.0 B | 32x32 | 824.523 us | 0.001 | 0.001 |
| 128.0 B | 64x64 | 818.372 us | 0.003 | 0.002 |
| 256.0 B | 128x128 | 812.340 us | 0.005 | 0.004 |
| 512.0 B | 256x256 | 813.651 us | 0.010 | 0.008 |
| 1.0 KB | 512x512 | 833.654 us | 0.020 | 0.015 |
| 2.0 KB | 1024x1024 | 863.838 us | 0.038 | 0.028 |
| 4.0 KB | 2048x2048 | 921.941 us | 0.071 | 0.053 |
| 8.0 KB | 4096x4096 | 1.134 ms | 0.116 | 0.087 |
| 16.0 KB | 8192x8192 | 965.214 us | 0.272 | 0.204 |
| 32.0 KB | 16384x16384 | 1.081 ms | 0.485 | 0.364 |
| 64.0 KB | 32768x32768 | 1.899 ms | 0.552 | 0.414 |
| 128.0 KB | 65536x65536 | 1.660 ms | 1.264 | 0.948 |
| 256.0 KB | 131072x131072 | 2.405 ms | 1.744 | 1.308 |
| 512.0 KB | 262144x262144 | 3.387 ms | 2.477 | 1.857 |
| 1.0 MB | 524288x524288 | 9.475 ms | 1.771 | 1.328 |
| 2.0 MB | 1048576x1048576 | 17.951 ms | 1.869 | 1.402 |
| 4.0 MB | 2097152x2097152 | 51.344 ms | 1.307 | 0.980 |
| 8.0 MB | 4194304x4194304 | 121.307 ms | 1.106 | 0.830 |
| 16.0 MB | 8388608x8388608 | 225.367 ms | 1.191 | 0.893 |
| 32.0 MB | 16777216x16777216 | 569.948 ms | 0.942 | 0.706 |
| 64.0 MB | 33554432x33554432 | 1132.699 ms | 0.948 | 0.711 |
