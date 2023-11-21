#include <mpi.h>

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
int me, num;

//Считывает из файла размер матрицы
int visa(const char * a) {
  int n=2;
  ifstream ff(a);
  ff >> n;
  ff.close();
  return n;
}

//Генерация матрицы по формуле, либо считывание матрицы из файла
void Input(int skolko, double **a, int otkuda, const char * name) {
  int i,j,n;
  double p;
  //Строки будут равномерно распределены среди потоков. Причем в каждый поток попадают подряд
  //идущие строки
  if (otkuda == 1) {
    n = skolko;
    for (i = 0; i < n/num; i++)
      for (j = 0; j < n; j++)
        //В формуле учитываем, что для потока строки начинаются с нулевой, хотя
        //в оригинальной матрице эта строка (i+me*n/num)-ая
        a[i][j] = min(n-j,n-i-me*n/num);//1.0/(1.0+i+me*n/num+j);
  }
  if(otkuda==2) {
    ifstream f(name);
    f >> n;
    //До своего блока поток просто игнорирует строки
    for(i=0;i<me*n/num;i++)
      for(j=0;j<n;j++)
        f>>p;
    //Свой блок поток сохраняет себе в память
    for (i = 0; i < n/num; i++)
      for (j = 0; j < n; j++)
        f >> a[i][j];
    f.close();
  }
}

//Многопоточный вывод матрицы на экран
void Output(int n, double **a) {
  int i,j,k = 0;
  // output only first 6 rows & cols
  for(i=0;i<min(n,6);i++) {
    //Синхронизируем все потоки перед выводом очередной строки
    MPI_Barrier(MPI_COMM_WORLD);
    //Так как как мы распределили строки равномерно между потоками, причем в каждом потоке подряд идущие
    //строки, то сначала выводит свои строки первый поток. Если у него меньше чем 6 строк, то выводит второй
    //поток. И так далее
    if(me==k) {
      for(j=0;j<min(n,6);j++)
        cout<<setw(12)<< a[i-me*n/num][j];
      cout << endl;
    }
    if(i-k*n/num==n/num-1) k++;
  }
}

//Сведение оригинальной матрицы к диагональному виду и построение обратной из присоединенной матрицы.
//Обмен информацией между потоками занимает O(n) памяти, где n - размер матрицы. функция зануляет
//столбец под номером k (кроме диагонального элемента) с выбором главного элемента по строке
int Inv(int n, double **a,double **x,int k) {
  //Массивы, с помощью который будет передваться информация между потоками
  double *ms;
  ms = new double[n];
  double *mf;
  mf = new double[n];
  int i;

  //Каждый поток вставляет в массив ms значения своего столбца, который диагонализруется в данный момент. На
  //остальный места каждый поток вставляет ноль
  for(i=0;i<n;i++) ms[i] = 0.0;
  for(i=me*n/num;i<(me+1)*n/num;i++) ms[i] = a[i-me*n/num][k];

  //Потоки суммируют полученные столбцы ms в столбец mf. Получится, что столбец mf совпадает с рассматриваемым
  //столбцом
  MPI_Barrier(MPI_COMM_WORLD);
  for (i=0;i<num;i++)
    MPI_Reduce(ms, mf, n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  //Ищем главный элемент в столбце
  int M = k;
  for(i=k+1;i<n;i++)
    if(fabs(mf[i]) > fabs(mf[M]))
      M = i;

  //Если главный элемент равен нулю, то у матрицы нет обратной
  if(fabs(mf[M])<1e-7) {
    delete[]ms;
    delete[]mf;
    return -1;
  }

  //Освобождаем память
  delete[]ms;
  delete[]mf;

  int t1,t2;
  t1 = k/(n/num);//номер потока, где содержится строка с тем же номером, что столбец, который мы диагонализируем
  t2 = M/(n/num);//номер потока с главным элементом

  //Новые массивы для обмена информацией между потоками
  double *s1;
  s1 = new double[4*n];
  double *s2;
  s2 = new double[4*n];

  //Все потоки зануляют созданные массивы
  for(i=0;i<4*n;i++) {
    s1[i]=0.0;
  }

  //Поток, содержащий элемент на диагонали, заносит в массивы соответсвующие строки
  if(me==t1)
    for(i=n;i<2*n;i++) {
      s1[i] = a[k-me*n/num][i-n];
      s1[i+2*n] = x[k-me*n/num][i-n];
    }

  //Поток с главным элементом заносит в массивы соответсвующие строки
  if(me==t2)
    for(i=0;i<n;i++) {
      s1[i] = a[M-me*n/num][i];
      s1[i+2*n] = x[M-me*n/num][i];
    }

  //Снова суммируем все массивы в массив s2
  MPI_Barrier(MPI_COMM_WORLD);
  for (i=0;i<num;i++)
    MPI_Reduce(s1, s2, 4*n, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  //Все потоки нормируют строку с главным элементом
  double c = s2[k];
  for(i=0;i<n;i++) {
    s2[i] = s2[i]/c;
    s2[i+2*n] = s2[i+2*n]/c;
  }

  //Потоки с главным элементом и элементом на диагонали перестанавливают строки
  if(me==t2)
    for(i=n;i<2*n;i++) {
      a[M-me*n/num][i-n] = s2[i];
      x[M-me*n/num][i-n] = s2[i+2*n];
    }
  if(me==t1)
    for(i=0;i<n;i++) {
      a[k-me*n/num][i] = s2[i];
      x[k-me*n/num][i] = s2[i+2*n];
    }

  //Теперь все потоки отнимают строку на диагонали от своих строк, тем самым зануляя
  //столбец, с которым мы работаем (кроме диагонального элемента)
  int j;
  for(i=0;i<n/num;i++) {
    if(i+me*n/num!=k) {
      c = a[i][k];
      for(j=0;j<n;j++) {
        a[i][j] = a[i][j]-s2[j]*c;
        x[i][j] = x[i][j]-s2[j+2*n]*c;
      }
    }
  }
  delete[]s1;
  delete[]s2;
  return 1;
}

int main(int argc, char * argv[]) {
  // Каждый поток инициализирует свой номер и число потоков
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int i;
  int n;
  double **a;
  double **x;
  ifstream input;
  double startwtime = 0.0;
  double endwtime;
  int t, j;
  unsigned int skolko = 2, otkuda = 4;

  // В этот раз обойдемся без случайной генерации матрицы
  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-' && argv[i][1] == 'm' && argv[i][2] == '\0')
      otkuda = 1;
    if (argv[i][0] == '-' && argv[i][1] == 'f' && argv[i][2] == '\0')
      otkuda = 2;
  }
  if (otkuda == 4) {
    cout << "-m <size> - genereate by formulae\n"
         << "-f <filepath> - get from file\n";
    return -1;
  }

  // Если генерируем по формуле, то считываем размер генерируемой матрицы. Функцией atoi по условию было пользоваться запрещено
  if (otkuda == 1) {
    for (i = 1; i < argc; i++) {
      t = 0;
      skolko = 0;
      for (j = 0; j < (int)strlen(argv[i]); j++) {
        skolko = skolko * 10 + int(argv[i][j]) - 48;
        if (int(argv[i][j]) < 48 || int(argv[i][j]) > 57)
          t = 1;
      }
      if (t == 0)
        break;
    }
    if (t == 1)
      skolko = 2;
    n = skolko;
  }

  // Если считываем из файла, то пытаемся открыть его
  if (otkuda == 2) {
    t = 0;
    for (i = 1; i < argc; i++) {
      if (ifstream(argv[i])) {
        skolko = i;
        n = visa(argv[i]);
        t = 1;
      }
    }
    if (t == 0) {
      cout << "Ne ukazan fail" << endl;
      return -2;
    }
  }

  // Итак, равномерно распределяем строки между потоками
  a = new double *[n / num];
  for (i = 0; i < n / num; i++)
    a[i] = new double[n];
  x = new double *[n / num];
  for (i = 0; i < n / num; i++)
    x[i] = new double[n];

  // Каждый поток заполняет свою часть присоединенной матрицы
  for (i = 0; i < n / num; i++)
    for (j = 0; j < n; j++)
      if (i + me * n / num == j)
        x[i][j] = 1.0;
      else
        x[i][j] = 0.0;

  // Проверяем, чтобы всем хватило памяти
  if (!(a && x)) {
    cout << "Not enough memory!" << endl;

    if (a) {
      for (i = 0; i < n / num; i++)
        delete[] a[i];
      delete[] a;
    }
    if (x) {
      for (i = 0; i < n / num; i++)
        delete[] x[i];
      delete[] x;
    }

    return -3;
  }

  // Заполняем оригинальную матрицу
  Input(skolko, a, otkuda, argv[min(skolko, 100 * (otkuda - 1) * otkuda)]);

  // Синхронизируемся и засекаем время
  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) {
    startwtime = MPI_Wtime();
  }

  // Запускаем поиск обратной матрицы. Поочередно каждый столбец приводим к диагональному виду
  for (i = 0; i < n; i++) {
    j = Inv(n, a, x, i);
    if (j == -1) {
      cout << "Net obratnoi" << endl;
      for (i = 0; i < n / num; i++)
        delete[] a[i];
      delete[] a;
      for (i = 0; i < n / num; i++)
        delete[] x[i];
      delete[] x;
      return -4;
    }
  }

  // Останавливаем таймер
  if (me == 0) {
    endwtime = MPI_Wtime();
  }

  //MPI_Barrier(MPI_COMM_WORLD);
  //Output(n, x);
  MPI_Barrier(MPI_COMM_WORLD);
  if(me==0) {
    cout << "elapsed time: " << (double)(endwtime-startwtime)<<" s\n";
  }

  for(i=0;i<n/num;i++) delete[]a[i];
  delete[]a;
  for(i=0;i<n/num;i++) delete[]x[i];
  delete[]x;
  MPI_Finalize();
  return 0;
}