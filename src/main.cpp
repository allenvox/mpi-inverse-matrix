#include <mpi.h>

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>

const std::string prefix = "[inverse matrix] ";

int n, nrows, rank, commsize;

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
  std::cout << prefix << rank << ": lb = " << *lb << ", ub = " << *ub << '\n';
}

double** get_matrix() {
  double **a = new double *[nrows];
  for (int i = 0; i < nrows; i++) {
    a[i] = new double[n];
    for (int j = 0; j < n; j++) {
      a[i][j] = std::min(n - j, n - i - rank * n / commsize);
    }
  }
  return a;
}

void print_matrix(double **a) {
  int k = 0;
  // output only first 6 rows & cols
  for (int i = 0; i < std::min(nrows, 6); i++) {
    // sync all procs before outputting next row
    MPI_Barrier(MPI_COMM_WORLD);
    // each proc outputs his own rows
    if (rank == k) {
      for (int j = 0; j < std::min(nrows, 6); j++) {
        std::cout << std::setw(12) << a[i - rank * nrows][j];
      }
      std::cout << '\n';
    }
    if (i - k * nrows == nrows - 1) {
      k++;
    }
  }
}

// diagonalising matrix & getting inverse from connected matrix
// transferring data between procs takes O(n) memspace
// function zeroes k-th col except diagonal elem, choosing row's main elem
void inverse_matrix(double **a, double **x, int k, int lb) {
  double local_max = 0.0;
  int local_index = -1;
  for (int i = 0; i < nrows; i++) {
    int global_index = i + lb;
    if (global_index >= k && std::fabs(a[i][k]) > local_max) {
      local_max = std::fabs(a[i][k]);
      local_index = global_index;
    }
  }

  // Используем MPI_Reduce для определения глобального максимума и его индекса
  struct {
    double value;
    int index;
  } local_data = {local_max, local_index}, global_data;

  MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  if (global_data.value == 0.0) {
    throw std::runtime_error(std::to_string(rank) + ": No inverse matrix\n");
  }

  /*
  // for transferring data between procs
  double *ms = new double[n];
  double *mf = new double[n];
  // each proc inserts his diagonalised col values to ms, other = 0
  for (int i = 0; i < n; i++) {
    ms[i] = 0.0;
  }
  for (int i = rank * nrows; i < (rank + 1) * nrows; i++) {
    ms[i] = a[i - rank * nrows][k];
  }

  // procs sum ms cols to mf col => mf = current col
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < commsize; i++) {
    MPI_Reduce(ms, mf, n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // find main element in col
  int M = k;
  for (int i = k + 1; i < n; i++) {
    if (std::fabs(mf[i]) > std::fabs(mf[M])) {
      M = i;
    }
  }
  delete[] ms;
  delete[] mf;

  // if main element is 0, matrix got no inverse
  if (std::fabs(mf[M]) == 0.0) {
    return -1;
  }
   */

  // t1 - proc with n-th row, n is diagonalised col number
  // t2 - proc with main element
  int M = global_data.index;
  int t1 = k / (nrows);
  int t2 = M / (nrows);

  // new transferring data
  double *s1 = new double[4 * n];
  double *s2 = new double[4 * n];
  for (int i = 0; i < 4 * n; i++) {
    s1[i] = 0.0;
  }

  // proc with diagonal element puts needed rows to arrays
  if (rank == t1) {
    for (int i = n; i < 2 * n; i++) {
      s1[i] = a[k - rank * nrows][i - n];
      s1[i + 2 * n] = x[k - rank * nrows][i - n];
    }
  }

  // proc with main elem puts needed rows to arrays
  if (rank == t2) {
    for (int i = 0; i < n; i++) {
      s1[i] = a[M - rank * nrows][i];
      s1[i + 2 * n] = x[M - rank * nrows][i];
    }
  }

  // sum all arrays to s2
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < commsize; i++) {
    MPI_Reduce(s1, s2, 4 * n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // all procs normalise row with main elem
  double c = s2[k];
  for (int i = 0; i < n; i++) {
    s2[i] = s2[i] / c;
    s2[i + 2 * n] = s2[i + 2 * n] / c;
  }

  // procs with main and diagonal elements swap rows
  if (rank == t2) {
    for (int i = n; i < 2 * n; i++) {
      a[M - rank * nrows][i - n] = s2[i];
      x[M - rank * nrows][i - n] = s2[i + 2 * n];
    }
  }
  if (rank == t1) {
    for (int i = 0; i < n; i++) {
      a[k - rank * nrows][i] = s2[i];
      x[k - rank * nrows][i] = s2[i + 2 * n];
    }
  }

  // all procs subtract diagonal row from their rows
  // (zeroing working col except diagonal elem)
  for (int i = 0; i < nrows; i++) {
    if (i + rank * nrows != k) {
      c = a[i][k];
      for (int j = 0; j < n; j++) {
        a[i][j] = a[i][j] - s2[j] * c;
        x[i][j] = x[i][j] - s2[j + 2 * n] * c;
      }
    }
  }
  delete[] s1;
  delete[] s2;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc > 1) {
    n = std::atoi(argv[1]);
  } else {
    n = 1000;
  }

  // each proc gets equal amount of rows
  int lb, ub;
  get_chunk(&lb, &ub);
  nrows = ub - lb + 1;

  double **a = get_matrix();
  double **x = new double*[nrows];
  for (int i = 0; i < nrows; i++) {
    x[i] = new double[n];
  }

  // each proc fills its chunk of connected matrix
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < n; j++) {
      if (i + rank * nrows == j) {
        x[i][j] = 1.0;
      } else {
        x[i][j] = 0.0;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t = -MPI_Wtime();

  for (int i = 0; i < n; i++) { // find inverse matrix, diagonalise each col
    try {
      inverse_matrix(a, x, i, lb);
    }
    catch(const std::exception &e) {
      for (int j = 0; j < nrows; j++) {
        delete[] a[j];
        delete[] x[j];
      }
      delete[] a;
      delete[] x;
      std::cerr << "error: " << e.what() << '\n';
      return 1;
    }
  }

  t += MPI_Wtime();
  if (rank == 0) {
    std::cout << prefix << "commsize = " << commsize << ", n = " << n
              << ", t = " << t << " sec\n";
  }

  for (int i = 0; i < nrows; i++) {
    delete[] a[i];
    delete[] x[i];
  }
  delete[] a;
  delete[] x;
  MPI_Finalize();
  return 0;
}