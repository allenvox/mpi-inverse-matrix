#include <mpi.h>

#include <cmath>
#include <iomanip>
#include <iostream>

int n, rank, commsize;

void get_matrix(double **a) {
  // each proc gets equal amount of continuous rows, last proc gets remaining
  for (int i = 0; i < n / commsize; i++) {
    for (int j = 0; j < n; j++) {
      // proc's n-th row = matrix's (i + rank * n / commsize)-th row
      a[i][j] = std::min(n - j, n - i - rank * n / commsize);
    }
  }
}

void print_matrix(double **a) {
  int k = 0;
  // output only first 6 rows & cols
  for (int i = 0; i < std::min(n, 6); i++) {
    // sync all procs before outputting next row
    MPI_Barrier(MPI_COMM_WORLD);
    // each proc outputs his own rows
    if (rank == k) {
      for (int j = 0; j < std::min(n, 6); j++) {
        std::cout << std::setw(12) << a[i - rank * n / commsize][j];
      }
      std::cout << '\n';
    }
    if (i - k * n / commsize == n / commsize - 1) {
      k++;
    }
  }
}

// diagonalising matrix & getting inverse from connected matrix
// transferring data between procs takes O(n) memspace
// function zeroes k-th col except diagonal elem, choosing row's main elem
int inverse_matrix(double **a, double **x, int k) {
  // for transferring data between procs
  double *ms = new double[n];
  double *mf = new double[n];
  // each proc inserts his diagonalised col values to ms, other = 0
  for (int i = 0; i < n; i++) {
    ms[i] = 0.0;
  }
  for (int i = rank * n / commsize; i < (rank + 1) * n / commsize; i++) {
    ms[i] = a[i - rank * n / commsize][k];
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
  if (std::fabs(mf[M]) == 0) {
    return -1;
  }

  // t1 - proc with n-th row, n is diagonalised col number
  // t2 - proc with main element
  int t1 = k / (n / commsize);
  int t2 = M / (n / commsize);

  // new transferring data
  double *s1 = new double[4 * n];
  double *s2 = new double[4 * n];
  for (int i = 0; i < 4 * n; i++) {
    s1[i] = 0.0;
  }

  // proc with diagonal element puts needed rows to arrays
  if (rank == t1) {
    for (int i = n; i < 2 * n; i++) {
      s1[i] = a[k - rank * n / commsize][i - n];
      s1[i + 2 * n] = x[k - rank * n / commsize][i - n];
    }
  }

  // proc with main elem puts needed rows to arrays
  if (rank == t2) {
    for (int i = 0; i < n; i++) {
      s1[i] = a[M - rank * n / commsize][i];
      s1[i + 2 * n] = x[M - rank * n / commsize][i];
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
      a[M - rank * n / commsize][i - n] = s2[i];
      x[M - rank * n / commsize][i - n] = s2[i + 2 * n];
    }
  }
  if (rank == t1) {
    for (int i = 0; i < n; i++) {
      a[k - rank * n / commsize][i] = s2[i];
      x[k - rank * n / commsize][i] = s2[i + 2 * n];
    }
  }

  // all procs subtract diagonal row from their rows
  // (zeroing working col except diagonal elem)
  for (int i = 0; i < n / commsize; i++) {
    if (i + rank * n / commsize != k) {
      c = a[i][k];
      for (int j = 0; j < n; j++) {
        a[i][j] = a[i][j] - s2[j] * c;
        x[i][j] = x[i][j] - s2[j + 2 * n] * c;
      }
    }
  }
  delete[] s1;
  delete[] s2;
  return 1;
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
  double **a = new double *[n / commsize];
  double **x = new double *[n / commsize];
  for (int i = 0; i < n / commsize; i++) {
    a[i] = new double[n];
    x[i] = new double[n];
  }
  // each proc fills its chunk of connected matrix
  for (int i = 0; i < n / commsize; i++) {
    for (int j = 0; j < n; j++) {
      if (i + rank * n / commsize == j) {
        x[i][j] = 1.0;
      } else {
        x[i][j] = 0.0;
      }
    }
  }

  get_matrix(a);

  MPI_Barrier(MPI_COMM_WORLD);
  double t = -MPI_Wtime();

  for (int i = 0; i < n; i++) { // find inverse matrix, diagonalise each col
    int result = inverse_matrix(a, x, i);
    if (result == -1) {
      std::cout << "No inverse matrix\n";
      for (int j = 0; j < n / commsize; j++) {
        delete[] a[j];
        delete[] x[j];
      }
      delete[] a;
      delete[] x;
      return -4;
    }
  }

  t += MPI_Wtime();
  if (rank == 0) {
    std::cout << "n = " << n << ", t = " << t << " sec\n";
  }

  for (int i = 0; i < n / commsize; i++) {
    delete[] a[i];
    delete[] x[i];
  }
  delete[] a;
  delete[] x;
  MPI_Finalize();
  return 0;
}