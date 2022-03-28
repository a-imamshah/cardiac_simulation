#ifndef __CARDIACSIM_KERNELS__
#define __CARDIACSIM_KERNELS__

void simulate_version1(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b);

void simulate_version2(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b);

void simulate_version3(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b);

void simulate_version4(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b);

#endif
