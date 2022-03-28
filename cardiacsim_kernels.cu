/*

	Implement your CUDA kernel in this file

*/
#include <stdio.h>
#define TILE_DIM 32 


// adding ghose cells or padding
__global__ void ghost_cells(double *E_prev, const int n, const int m)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int N = n+2;
  // int M = m+2;

  if( row < n+1 && col < m+1)
  {
    int currIndex = row * N + col;
    if (row == 2 )
      E_prev[col] = E_prev[currIndex];
    if (row == n - 1 )
      E_prev[(m+1)*N+col] = E_prev[currIndex];
    if (col == 2)
      E_prev[row*N] = E_prev[currIndex];
    if (col == m - 1)
      E_prev[row*N+(N-1)] = E_prev[currIndex];

    __syncthreads();
  }
}

// version 1 PDE function
__global__ void pde_E(double *E, double *E_prev, const double alpha, const int n, const int m)
{            
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int N = n+2;

    if( row < n+1 && col < m+1)
    {
      int currIndex = row * N + col;
      int upIndex = currIndex - N;
      int downIndex = currIndex + N;
      int rightIndex = currIndex+1;
      int leftIndex = currIndex-1;

      E[currIndex] = E_prev[currIndex]+alpha*(E_prev[rightIndex]+E_prev[leftIndex]+ E_prev[downIndex]+ E_prev[upIndex]+(-4*E_prev[currIndex]));
  }
}


// version 1 ODE function
__global__ void ode_E(double *E, double *R, const int n, const int m, const double kk, const double dt, const double a){

  int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int col = blockIdx.x * blockDim.x + threadIdx.x + 1 ;
  int N = n+2;  
  if( row < n+1 && col < m+1)
  {
    int currIndex = row * N + col;
    E[currIndex] = E[currIndex] -dt * (kk * E[currIndex] * (E[currIndex] - a) * (E[currIndex]-1)+ E[currIndex] * R[currIndex]);
  }   
}

// version 1 ODE function
__global__ void ode_R(double *E, double *R, const int n, const int m, const double kk,
  const double dt, const double epsilon, const double M1,const double  M2, const double b)
  {
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1 ;
    int N = n+2;  
    if( col < n+1 && row < m+1)
    {
      int currIndex = row * N + col;
      R[currIndex] = R[currIndex] + dt * (epsilon + M1 * R[currIndex] / ( E[currIndex] + M2)) * (- R[currIndex] - kk * E[currIndex] * (E[currIndex] - b - 1));
    }   
  }


// version 2 kernel function
__global__ void kernel_version2(double *E, double *E_prev, double *R, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b)
  {
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1 ;
    int N = n+2;   
    if( row < n+1 && col < m+1){
  
      int currIndex = row * N + col;
      int upIndex = currIndex - N;
      int downIndex = currIndex + N;
      int rightIndex = currIndex+1;
      int leftIndex = currIndex-1;

      if (row == 2 )
        E_prev[col] = E_prev[currIndex];
      if (row == n - 1 )
        E_prev[(m+1)*N+col] = E_prev[currIndex];
      if (col == 2)
        E_prev[row*N] = E_prev[currIndex];
      if (col == m - 1)
        E_prev[row*N+(N-1)] = E_prev[currIndex];

      __syncthreads();
      E[currIndex] = E_prev[currIndex]+alpha*(E_prev[rightIndex]+E_prev[leftIndex]+ E_prev[downIndex]+ E_prev[upIndex]-4 * E_prev[currIndex]);
      E[currIndex] = E[currIndex] -dt * (kk * E[currIndex] * (E[currIndex] - a) * (E[currIndex]-1)+ E[currIndex] * R[currIndex]);
      R[currIndex] = R[currIndex] + dt * (epsilon + M1 * R[currIndex] / ( E[currIndex] + M2)) * (- R[currIndex] - kk * E[currIndex] * (E[currIndex] - b - 1));

    }
  }

// version 3 kernel function
__global__ void kernel_version3(double *E, double *E_prev, double *R, const double alpha, const int n, const int m, const double kk,
        const double dt, const double a, const double epsilon,
        const double M1,const double  M2, const double b)
        {

          int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
          int col = blockIdx.x * blockDim.x + threadIdx.x + 1 ;
          int N = n+2;   
          if( row < n+1 && col < m+1){
        
            int currIndex = row * N + col;
            int upIndex = currIndex - N;
            int downIndex = currIndex + N;
            int rightIndex = currIndex+1;
            int leftIndex = currIndex-1;

            if (row == 2 )
              E_prev[col] = E_prev[currIndex];
            if (row == n - 1 )
              E_prev[(m+1)*N+col] = E_prev[currIndex];
            if (col == 2)
              E_prev[row*N] = E_prev[currIndex];
            if (col == m - 1)
              E_prev[row*N+(N-1)] = E_prev[currIndex];

            __syncthreads();
      
            E[currIndex] = E_prev[currIndex]+alpha*(E_prev[rightIndex]+E_prev[leftIndex]+ E_prev[downIndex]+ E_prev[upIndex] +(-4*E_prev[currIndex]));
            double e = E[currIndex];
            double r = R[currIndex];

            e = e -dt * (kk * e * (e - a) * (e-1)+ e * r);
            r  = r + dt * (epsilon + M1 * r / ( e + M2)) * (- r - kk * e * (e - b - 1));

            E[currIndex] = e;
            R[currIndex] = r;
          }
        }


// version 4 kernel function
__global__ void kernel_version4(double *E, double *E_prev, double *R, const double alpha, const int n, const int m, const double kk,
        const double dt, const double a, const double epsilon,
        const double M1,const double  M2, const double b){

      __shared__ double tile_Eprev[TILE_DIM+2][TILE_DIM+2];

      int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
      int col = blockIdx.x * blockDim.x + threadIdx.x + 1 ;
      int N = n+2; 
      
      int l_row = threadIdx.y + 1;
      int l_col = threadIdx.x + 1;

      if( col < n+1 && row < m+1)
      {
        int currIndex = row * N + col;
        if (row == 2 )
          E_prev[col] = E_prev[currIndex];
        if (row == n - 1 )
          E_prev[(n+1)*N+col] = E_prev[currIndex];
        if (col == 2)
          E_prev[row*N] = E_prev[currIndex];
        if (col == m - 1)
          E_prev[row*N+(m+1)] = E_prev[currIndex];

        __syncthreads();
        
        tile_Eprev[l_row][l_col] = E_prev[currIndex];
        
        if(l_row == 1)
          tile_Eprev[l_row -1][l_col] = E_prev[currIndex-N];
        if(l_row == TILE_DIM)
          tile_Eprev[l_row+1][l_col] = E_prev[currIndex+N];
        if (l_col == 1)
          tile_Eprev[l_row][l_col-1] = E_prev[currIndex-1];
        if (l_col == TILE_DIM)
          tile_Eprev[l_row][l_col+1] = E_prev[currIndex+1];

        __syncthreads();

        E[currIndex] = tile_Eprev[l_row][l_col]+alpha*(tile_Eprev[l_row][l_col+1]+tile_Eprev[l_row][l_col-1]+ tile_Eprev[l_row+1][l_col]+ tile_Eprev[l_row-1][l_col]-4 * tile_Eprev[l_row][l_col]);
        double e = E[currIndex];
        double r = R[currIndex];
        e = e -dt * (kk * e * (e - a) * (e-1)+ e * r);
        r  = r + dt * (epsilon + M1 * r / ( e + M2)) * (- r - kk * e * (e - b - 1));
        E[currIndex] = e;
        R[currIndex] = r;

      }   
    }
  


void simulate_version1(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b) {

  const dim3 block_size(bx,by);
  const dim3 num_blocks((n+block_size.x-1) / block_size.x, (m+block_size.y-1) / block_size.y);

    ghost_cells<<<num_blocks,block_size>>>(E_prev, n, m);
    pde_E<<<num_blocks,block_size>>>(E, E_prev, alpha, n, m);
    ode_E<<<num_blocks,block_size>>>(E, R, n, m, kk, dt, a);
    ode_R<<<num_blocks,block_size>>>(E, R, n, m, kk, dt, epsilon, M1, M2, b);

}

void simulate_version2(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b) {

  const dim3 block_size(bx,by);
  const dim3 num_blocks((n+block_size.x-1) / block_size.x, (m+block_size.y-1) / block_size.y);

  kernel_version2<<<num_blocks,block_size>>>(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
}

void simulate_version3(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b) {

  const dim3 block_size(bx,by);
  const dim3 num_blocks((n+block_size.x-1) / block_size.x, (m+block_size.y-1) / block_size.y);

  kernel_version3<<<num_blocks,block_size>>>(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
}

void simulate_version4(double *E, double *E_prev, double *R, const int bx, const int by, const double alpha, const int n, const int m, const double kk,
  const double dt, const double a, const double epsilon,
  const double M1,const double  M2, const double b) {

  const dim3 block_size(bx,by);
  const dim3 num_blocks((n+block_size.x-1) / block_size.x, (m+block_size.y-1) / block_size.y);

  kernel_version4<<<num_blocks,block_size>>>(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
}