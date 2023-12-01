#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using std::cout;

int n = 1680, nrows, rank, commsize;

void get_chunk(int *lb, int *ub) {
  int rows_per_process = n / commsize;
  int remaining_rows = n % commsize;
  // Determine the lower and upper bounds for each process
  if (rank < remaining_rows) {
    *lb = rank * (rows_per_process + 1);
    *ub = *lb + rows_per_process;
  } else {
    *lb = remaining_rows * (rows_per_process + 1) + (rank - remaining_rows) * rows_per_process;
    *ub = *lb + rows_per_process - 1;
  }
  cout << rank << ": lb = " << *lb << ", ub = " << *ub << '\n';
}

double* get_input_matrix() {
  double *a = (double*)malloc(nrows * n * sizeof(double));
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < n; j++) {
      a[i * n + j] = std::min(n - j, n - i - rank * nrows);
    }
  }
  return a;
}

double* get_connected_matrix() {
  double *x = (double*)malloc(nrows * n * sizeof(double));
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < n; j++) {
      if (i + rank * nrows == j) {
        x[i * n + j] = 1.0;
      } else {
        x[i * n + j] = 0.0;
      }
    }
  }
  return x;
}

// diagonalising matrix & getting inverse from connected matrix
// transferring data between procs takes O(n) memspace
// function zeroes k-th col except diagonal elem, choosing row's main elem
void inverse_matrix(double *a, double *x, int k, int lb) {
  double local_max = 0.0;
  int local_index = -1;
  for (int i = 0; i < nrows; i++) {
    int global_index = i + lb;
    if (global_index >= k && std::fabs(a[i * n + k]) > local_max) {
      local_max = std::fabs(a[i * n + k]);
      local_index = global_index;
    }
  }

  // определения глобального максимума и его индекса
  struct {
    double value;
    int index;
  } local_data = {local_max, local_index}, global_data;

  MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  if (global_data.value == 0.0) { // if global max is 0
    std::cerr << std::to_string(rank) << " no inverse matrix\n";
  }

  int M = global_data.index;
  int t1 = k / (nrows); // proc with n-th row, n is diagonalised col number
  int t2 = M / (nrows); // proc with main elem

  // new transferring data
  double *s1 = new double[4 * n];
  double *s2 = new double[4 * n];
  for (int i = 0; i < 4 * n; i++) {
    s1[i] = 0.0;
  }

  // proc with diagonal element puts needed rows to arrays
  if (rank == t1) {
    for (int i = n; i < 2 * n; i++) {
      s1[i] = a[(k - rank * nrows) * n + i - n];
      s1[i + 2 * n] = x[(k - rank * nrows) * n + i - n];
    }
  }

  // proc with main elem puts needed rows to arrays
  if (rank == t2) {
    for (int i = 0; i < n; i++) {
      s1[i] = a[(M - rank * nrows) * n + i];
      s1[i + 2 * n] = x[(M - rank * nrows) * n + i];
    }
  }

  // sum all arrays to s2
  for (int i = 0; i < commsize; i++) {
    MPI_Reduce(s1, s2, 4 * n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  }

  // all procs normalise row with main elem
  double c = s2[k];
  for (int i = 0; i < n; i++) {
    s2[i] = s2[i] / c;
    s2[i + 2 * n] = s2[i + 2 * n] / c;
  }

  // procs with main and diagonal elements swap rows
  if (rank == t2) {
    for (int i = n; i < 2 * n; i++) {
      a[(M - rank * nrows) * n + i - n] = s2[i];
      x[(M - rank * nrows) * n + i - n] = s2[i + 2 * n];
    }
  }
  if (rank == t1) {
    for (int i = 0; i < n; i++) {
      a[(k - rank * nrows) * n + i] = s2[i];
      x[(k - rank * nrows) * n + i] = s2[i + 2 * n];
    }
  }

  // all procs subtract diagonal row from their rows
  // (zeroing working col except diagonal elem)
  for (int i = 0; i < nrows; i++) {
    if (i + rank * nrows != k) {
      c = a[i * n + k];
      for (int j = 0; j < n; j++) {
        a[i * n + j] = a[i * n + j] - s2[j] * c;
        x[i * n + j] = x[i * n + j] - s2[j + 2 * n] * c;
      }
    }
  }
  delete[] s1;
  delete[] s2;
}

int main(int argc, char **argv) {
  double t = -MPI_Wtime();
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }

  int lb, ub;
  get_chunk(&lb, &ub);
  nrows = ub - lb + 1;

  double *a = get_input_matrix();
  double *x = get_connected_matrix();

  //double t = -MPI_Wtime();
  for (int i = 0; i < n; ++i) {
    inverse_matrix(a, x, i, lb);
  }
  t += MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);
  cout << commsize << " procs, n = " << n << ", t = " << t << " sec\n";

  free(a);
  free(x);
  MPI_Finalize();
  return 0;
}