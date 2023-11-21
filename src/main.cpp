#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
int me, num;

enum { VOID_INPUT, FORMULA_INPUT, FILE_INPUT };

int size_from_file(const string &path) {
  int n = 2;
  ifstream ff(path);
  ff >> n;
  ff.close();
  return n;
}

void get_matrix(int skolko, double **a, int otkuda, const char *name) {
  int n;
  double p;
  // each proc gets equal amount of continuous rows
  if (otkuda == 1) {
    n = skolko;
    for (int i = 0; i < n / num; i++) {
      for (int j = 0; j < n; j++) {
        // proc's n-th row = matrix's (i + rank * n / commsize)-th row
        // 1 / (1 + i + rank * n / commsize + j)
        a[i][j] = min(n - j, n - i - me * n / num);
      }
    }
  }
  if (otkuda == 2) {
    ifstream f(name);
    f >> n;
    // ignore rows before his own rows
    for (int i = 0; i < me * n / num; i++) {
      for (int j = 0; j < n; j++) {
        f >> p;
      }
    }
    // get his own rows
    for (int i = 0; i < n / num; i++) {
      for (int j = 0; j < n; j++) {
        f >> a[i][j];
      }
    }
    f.close();
  }
}

void print_matrix(int n, double **a) {
  int k = 0;
  // output only first 6 rows & cols
  for (int i = 0; i < min(n, 6); i++) {
    // sync all procs before outputting next row
    MPI_Barrier(MPI_COMM_WORLD);
    // each proc outputs his own rows
    if (me == k) {
      for (int j = 0; j < min(n, 6); j++) {
        cout << setw(12) << a[i - me * n / num][j];
      }
      cout << '\n';
    }
    if (i - k * n / num == n / num - 1) {
      k++;
    }
  }
}

// diagonalising matrix & getting inverse from connected matrix
// transferring data between procs takes O(n) memspace
// function zeroes k-th col except diagonal elem, choosing main elem for row
int inverse_matrix(int n, double **a, double **x, int k) {
  // arrays for transferring data between procs
  double *ms = new double[n];
  double *mf = new double[n];
  // each proc inserts his diagonalised col values to ms, other = 0
  for (int i = 0; i < n; i++) {
    ms[i] = 0.0;
  }
  for (int i = me * n / num; i < (me + 1) * n / num; i++) {
    ms[i] = a[i - me * n / num][k];
  }

  // procs sum ms cols to mf col => mf = current col
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < num; i++) {
    MPI_Reduce(ms, mf, n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // find main element in col
  int M = k;
  for (int i = k + 1; i < n; i++) {
    if (fabs(mf[i]) > fabs(mf[M])) {
      M = i;
    }
  }
  delete[] ms;
  delete[] mf;

  // if main element is close to 0, matrix got no inverse
  if (fabs(mf[M]) < 1e-7)
    return -1;

  int t1 = k / (n / num); // proc w/ row that's number = diagonalised col number
  int t2 = M / (n / num); // proc w/ main elem

  // new arrays for transferring data
  double *s1 = new double[4 * n];
  double *s2 = new double[4 * n];
  for (int i = 0; i < 4 * n; i++) {
    s1[i] = 0.0;
  }

  // proc with diagonal element puts needed rows to arrays
  if (me == t1) {
    for (int i = n; i < 2 * n; i++) {
      s1[i] = a[k - me * n / num][i - n];
      s1[i + 2 * n] = x[k - me * n / num][i - n];
    }
  }

  // proc with main elem puts needed rows to arrays
  if (me == t2)
    for (int i = 0; i < n; i++) {
      s1[i] = a[M - me * n / num][i];
      s1[i + 2 * n] = x[M - me * n / num][i];
    }

  // sum all arrays to s2
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < num; i++) {
    MPI_Reduce(s1, s2, 4 * n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // all procs normalise row with main elem
  double c = s2[k];
  for (int i = 0; i < n; i++) {
    s2[i] = s2[i] / c;
    s2[i + 2 * n] = s2[i + 2 * n] / c;
  }

  // procs with main elem and diagonal elem swap rows
  if (me == t2) {
    for (int i = n; i < 2 * n; i++) {
      a[M - me * n / num][i - n] = s2[i];
      x[M - me * n / num][i - n] = s2[i + 2 * n];
    }
  }
  if (me == t1) {
    for (int i = 0; i < n; i++) {
      a[k - me * n / num][i] = s2[i];
      x[k - me * n / num][i] = s2[i + 2 * n];
    }
  }

  // all procs subtract diagonal row from their rows
  // (zeroing working col except diagonal elem)
  for (int i = 0; i < n / num; i++) {
    if (i + me * n / num != k) {
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
  MPI_Comm_size(MPI_COMM_WORLD, &num);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int n, t;
  ifstream input;
  double startwtime, endwtime;
  unsigned int skolko = 2, otkuda = 4;

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-' && argv[i][1] == 'm' && argv[i][2] == '\0') {
      otkuda = FORMULA_INPUT;
    }
    if (argv[i][0] == '-' && argv[i][1] == 'f' && argv[i][2] == '\0') {
      otkuda = FILE_INPUT;
    }
  }
  if (otkuda == 4) {
    cout << "-m <size> - generate by formula\n-f <filepath> - get from file\n";
    return -1;
  }

  if (otkuda == FORMULA_INPUT) {
    for (int i = 1; i < argc; i++) {
      t = 0;
      skolko = 0;
      for (int j = 0; j < (int)strlen(argv[i]); j++) {
        skolko = skolko * 10 + int(argv[i][j]) - 48;
        if (int(argv[i][j]) < 48 || int(argv[i][j]) > 57) {
          t = 1;
        }
      }
      if (t == 0) {
        break;
      }
    }
    if (t == 1) {
      skolko = 2;
    }
    n = skolko;
  }

  if (otkuda == FILE_INPUT) {
    t = 0;
    for (int i = 1; i < argc; i++) {
      if (ifstream(argv[i])) {
        skolko = i;
        n = size_from_file(argv[i]);
        t = 1;
      }
    }
    if (t == 0) {
      cout << "Filepath not specified\n";
      return -2;
    }
  }

  // each proc gets equal amount of rows
  double **a = new double *[n / num];
  double **x = new double *[n / num];
  for (int i = 0; i < n / num; i++) {
    a[i] = new double[n];
    x[i] = new double[n];
  }
  // each proc fills its chunk of connected matrix
  for (int i = 0; i < n / num; i++) {
    for (int j = 0; j < n; j++) {
      if (i + me * n / num == j) {
        x[i][j] = 1.0;
      } else {
        x[i][j] = 0.0;
      }
    }
  }

  if (!(a && x)) {
    cout << "Not enough memory!\n";
    if (a) {
      for (int i = 0; i < n / num; i++)
        delete[] a[i];
      delete[] a;
    }
    if (x) {
      for (int i = 0; i < n / num; i++)
        delete[] x[i];
      delete[] x;
    }
    return -3;
  }

  // get input matrix
  get_matrix(skolko, a, otkuda, argv[min(skolko, 100 * (otkuda - 1) * otkuda)]);

  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) {
    startwtime = MPI_Wtime();
  }

  // find inverse matrix, diagonalise each col
  for (int i = 0; i < n; i++) {
    int result = inverse_matrix(n, a, x, i);
    if (result == -1) {
      cout << "No inverse matrix\n";
      for (int j = 0; j < n / num; j++) {
        delete[] a[j];
        delete[] x[j];
      }
      delete[] a;
      delete[] x;
      return -4;
    }
  }

  if (me == 0) {
    endwtime = MPI_Wtime();
  }

  // MPI_Barrier(MPI_COMM_WORLD);
  // print_matrix(n, x);
  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) {
    cout << "elapsed " << endwtime - startwtime << " sec\n";
  }

  for (int i = 0; i < n / num; i++) {
    delete[] a[i];
    delete[] x[i];
  }
  delete[] a;
  delete[] x;
  MPI_Finalize();
  return 0;
}