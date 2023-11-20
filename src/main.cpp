#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

int n, commsize, rank;

double **get_input_matrix() {
  //Строки будут равномерно распределены среди потоков. Причем в каждый поток
  //попадают подряд идущие строки
  double **a = new double *[n / commsize];
  for (int i = 0; i < n / commsize; i++) {
    a[i] = new double[n];
  }
  for (int i = 0; i < n / commsize; i++) {
    for (int j = 0; j < n; j++) {
      // В формуле учитываем, что для потока строки начинаются с нулевой, хотя
      // в оригинальной матрице эта строка (i+me*n/num)-ая
      a[i][j] =
          min(n - j, n - i - rank * n / commsize); // 1.0/(1.0+i+me*n/num+j);
    }
  }
  return a;
}

// printing out the matrix
void print_matrix(double **a) {
  int k = 0;
  for (int i = 0; i < min(n, 6); i++) {
    MPI_Barrier(MPI_COMM_WORLD); // sync
    //Так как как мы распределили строки равномерно между потоками, причем в
    //каждом потоке подряд идущие строки, то сначала выводит свои строки первый
    //поток. Если у него меньше чем 6 строк, то выводит второй поток. И так
    //далее
    if (rank == k) {
      for (int j = 0; j < min(n, 6); j++) {
        cout << setw(12) << a[i - rank * n / commsize][j] << '\n';
      }
    }
    if (i - k * n / commsize == n / commsize - 1) {
      k++;
    }
  }
}

void delete_matrix(double **x) {
  for (int i = 0; i < n / commsize; i++) {
    delete[] x[i];
  }
  delete[] x;
}

//Сведение оригинальной матрицы к диагональному виду и построение обратной из
//присоединенной матрицы. Обмен информацией между потоками занимает O(n) памяти,
//где n - размер матрицы. функция зануляет столбец под номером k (кроме
//диагонального элемента) с выбором главного элемента по строке
int Inv(double **a, double **x, int k) {
  // arrays for transfering data between procs
  double *ms = new double[n];
  double *mf = new double[n];

  // create empty ms array
  for (int i = 0; i < n; i++) {
    ms[i] = 0.0;
  }
  // each proc inserts its currently diagonalised col to ms array
  for (int i = rank * n / commsize; i < (rank + 1) * n / commsize; i++) {
    ms[i] = a[i - rank * n / commsize][k];
  }

  // mf col = cur col, procs sum gotten cols to mf
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < commsize; i++) {
    MPI_Reduce(ms, mf, n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // find main element in a row
  int M = k;
  for (int i = k + 1; i < n; i++) {
    if (fabs(mf[i]) > fabs(mf[M])) {
      M = i;
    }
  }

  double main_elem = mf[M];
  delete[] ms;
  delete[] mf;
  // if main element is close to 0, no inverse matrix
  if (fabs(main_elem) < 1e-7) {
    return -1;
  }

  int t1 = k / (n / commsize); // proc that contains row same num as cur col
  int t2 = M / (n / commsize); // proc with main element

  // new arrays to transfer data between arrays
  double *s1 = new double[4 * n];
  double *s2 = new double[4 * n];
  for (int i = 0; i < 4 * n; i++) {
    s1[i] = 0.0;
  }

  //Поток, содержащий элемент на диагонали, заносит в массивы соответсвующие
  //строки
  if (rank == t1) {
    for (int i = n; i < 2 * n; i++) {
      s1[i] = a[k - rank * n / commsize][i - n];
      s1[i + 2 * n] = x[k - rank * n / commsize][i - n];
    }
  }

  // proc with main elem puts rows to array
  if (rank == t2) {
    for (int i = 0; i < n; i++) {
      s1[i] = a[M - rank * n / commsize][i];
      s1[i + 2 * n] = x[M - rank * n / commsize][i];
    }
  }

  // sum all s1s to s2
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

  // procs with main elem and diagonal elem swap rows
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

  //Теперь все потоки отнимают строку на диагонали от своих строк, тем самым
  //зануляя столбец, с которым мы работаем (кроме диагонального элемента)
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

//Функция перемножает матрицы а и х, отнимает от произведения единичную матрицу
//и считает квадратичную норму полученной матрицы. Здесь вычисления производятся
//на одном потоке, так как по условию эту часть распараллеливать не нужно было
double Norma(double **a, double **x, int n) {
  // each proc puts their piece into array
  double *s = new double[2 * n * n];
  double *f = new double[2 * n * n];
  int i, j;
  for (int i = 0; i < 2 * n * n; i++) {
    s[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n / commsize; j++) {
      s[n * (rank * (n / commsize) + j) + i] = a[j][i];
    }
  }
  for (i = 0; i < n; i++) {
    for (int j = 0; j < n / commsize; j++) {
      s[n * (rank * (n / commsize) + j) + i + n * n] = x[j][i];
    }
  }

  // all procs get full matrix
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < commsize; i++) {
    MPI_Reduce(s, f, 2 * n * n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  double **m1 = new double *[n];
  double **m2 = new double *[n];
  for (int i = 0; i < n; i++) {
    m1[i] = new double[n];
    m2[i] = new double[n];
  }
  for (int i = 0; i < n * n; i++) {
    m1[i / n][i % n] = f[i];
    m2[i / n][i % n] = f[i + n * n];
  }

  // norm
  double t, Norm = 0.0;
  int k;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      t = 0.0;
      for (int k = 0; k < n; k++) {
        t = t + m1[i][k] * m2[k][j];
      }
      if (i == j) {
        Norm = Norm + (t - 1.0) * (t - 1.0);
      } else {
        Norm = Norm + t * t;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    delete[] m1[i];
    delete[] m2[i];
  }
  delete[] m1;
  delete[] m2;
  delete[] s;
  delete[] f;

  return sqrt(Norm);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int t, i, j;
  int n = 2;
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

  // each proc gets their rows
  double **a = get_input_matrix(n);

  double **x = new double *[n / commsize];
  for (int i = 0; i < n / commsize; i++) {
      x[i] = new double[n];
  }
  // each proc fills their chunk of connected matrix
  for (int i = 0; i < n / commsize; i++) {
      for (int j = 0; j < n; j++) {
        if (i + rank * n / commsize == j) {
          x[i][j] = 1.0;
        } else {
          x[i][j] = 0.0;
        }
      }
  }

  if (rank == 0) {
    cout << "\nmatrix:\n\n";
  }
  print_matrix(a);

  // sync
  MPI_Barrier(MPI_COMM_WORLD);

  double time = -MPI_Wtime();
  // find inverse matrix, diagonalise each col
  for (int i = 0; i < n; i++) {
    j = Inv(n, a, x, i);
    if (j == -1) {
      cout << "No inverse matrix\n";
      delete_matrix(a);
      delete_matrix(x);
      return -4;
    }
  }
  time += MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);
  print_matrix(x);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    cout << "\nt: " << time << " s\n";
  }

  delete_matrix(a);
  delete_matrix(x);
  MPI_Finalize();
  return 0;
}
