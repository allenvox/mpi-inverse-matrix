# C++ MPI Inverse Matrix program
❗ Make sure you have a working MPI builder & executer (check by running `mpicxx & mpiexec` in terminal) ❗<br><br>
**Build: `make`<br>
Run: `mpiexec bin/main <N>`**<br>
(remember that matrix' size is **N x N**)<br><br>
### Notes
❗ **Biggest issue (unsolved)**: The program is only working successfully with matrix size that is a multiple of the number of processes<br><br>
**Examples:**<br>
if you have 2 processes, you can calculate inverse of a matrix for only even **N**<br>
if you have 4 processes, you can successfully run program only with **N** that is divisible by 4<br>
... and so on
