#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <vector>

using namespace std;

int n;

double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

bool invertMatrix(double *matrix) {
  double *temp = (double*)malloc(n * n * 2 * sizeof(double));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      temp[i * n + j] = matrix[i * n + j];
    }
    temp[i * n + i + n] = 1;
  }

  // direct way
  for (int i = 0; i < n; ++i) {
    // row normalization
    double diag = temp[i * n + i];
    if (diag == 0) return false; // no inverse matrix
    for (int j = 0; j < n * 2; ++j) {
      temp[i * n + j] /= diag;
    }
    // col zeroing
    for (int k = 0; k < n; ++k) {
      if (k == i) continue;
      double factor = temp[k * n + i];
      for (int j = 0; j < n * 2; ++j) {
        temp[k * n + j] -= factor * temp[i * n + j];
      }
    }
  }

  // getting result
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      matrix[i * n + j] = temp[i * n + j + n];
    }
  }
  return true;
}

double *get_matrix() {
  double *a = (double*)malloc(n * n * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i * n + j] = min(n - j, n - i);
    }
  }
  return a;
}

int main(int argc, char **argv) {
  double t = -wtime();
  if (argc > 1) {
    n = atoi(argv[1]);
  } else {
    n = 1680;
  }
  double *matrix = get_matrix();
  if (invertMatrix(matrix)) {
    t += wtime();
    cout << "n = " << n << ", t = " << t << " sec\n";
  } else {
    cout << "No inverse matrix\n";
  }
  free(matrix);
  return 0;
}
