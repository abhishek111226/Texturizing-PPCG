/*************************************************************************/
/* Lid Driven Cavity(LDC) - 3D simulation using LBM with a D3Q19 lattice */
/*                                                                       */
/* Scheme: 3-Grid, Pull, Compute, Keep                                   */
/*                                                                       */
/* Lattice naming convention                                             */
/*   http://arxiv.org/pdf/1202.6081.pdf                                  */
/*                                                                       */
/* Author: Irshad Pananilath <pmirshad+code@gmail.com>                   */
/*                                                                       */
/*************************************************************************/

#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#ifndef nX
#define nX 200  // Size of domain along x-axis
#endif

#ifndef nY
#define nY 200  // Size of domain along y-axis
#endif

#ifndef nZ
#define nZ 200  // Size of domain along z-axis
#endif

#ifndef nK
#define nK 19  // D3Q19
#endif

#ifndef nTimesteps
#define nTimesteps 1000  // Number of timesteps for the simulation
#endif

#define C 0
#define E 1
#define W 2
#define N 3
#define S 4
#define T 5
#define B 6
#define NE 7
#define NW 8
#define SE 9
#define SW 10
#define TE 11
#define TW 12
#define BE 13
#define BW 14
#define TN 15
#define TS 16
#define BN 17
#define BS 18

/* Tunable parameters */
const double re = 100;
const double uTopX = 0.05;
const double uTopY = 0.02;

/* Calculated parameters: Set in main */
double omega = 0.0;
double one_minus_omega = 0.0;

const double w1 = 1.0 / 3.0;
const double w2 = 1.0 / 18.0;
const double w3 = 1.0 / 36.0;

double grid[2][nZ][nY][nX][nK] __attribute__((aligned(32)));

void lbm_kernel(double, double, double, double, double, double, double, double,
                double, double, double, double, double, double, double, double,
                double, double, double, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, int, int, int, int);

void dumpVelocities(int timestep);
int timeval_subtract(struct timeval *result, struct timeval *x,
                     struct timeval *y);

int main(void) {
  int t = 0, z, y, x, k;
  double total_lattice_pts =
      (double)nZ * (double)nY * (double)nX * (double)nTimesteps;

  /* For timekeeping */
  int ts_return = -1;
  struct timeval start, end, result;
  double tdiff = 0.0;

  /* Compute values for global parameters */
  omega = 2.0 /
          ((6.0 * sqrt(uTopX * uTopX + uTopY * uTopY) * (nX - 0.5) / re) + 1.0);
  one_minus_omega = 1.0 - omega;

  printf(
      "3D Lid Driven Cavity simulation with D3Q19 lattice:\n"
      "\tfilename   : %s\n"
      "\tscheme     : 3-Grid, Fused, Pull\n"
      "\tgrid size  : %d x %d x %d = %.2lf * 10^3 Cells\n"
      "\tnTimeSteps : %d\n"
      "\tRe         : %.2lf\n"
      "\tuTopX      : %.6lf\n"
      "\tuTopY      : %.6lf\n"
      "\tomega      : %.6lf\n",
      __FILE__, nX, nY, nZ, nX * nY * nZ / 1.0e3, nTimesteps, re, uTopX, uTopY,
      omega);

  /* Initialize all 19 PDFs for each point in the domain to w1, w2 or w3
   * accordingly */
  for (z = 0; z < nZ; z++) {
    for (y = 0; y < nY; y++) {
      for (x = 0; x < nX; x++) {
        grid[0][z][y][x][0] = w1;
        grid[1][z][y][x][0] = w1;

        for (k = 1; k < 7; k++) {
          grid[0][z][y][x][k] = w2;
          grid[1][z][y][x][k] = w2;
        }

        for (k = 7; k < nK; k++) {
          grid[0][z][y][x][k] = w3;
          grid[1][z][y][x][k] = w3;
        }
      }
    }
  }

  /* To satisfy PET */
  short _nX = nX;
  short _nY = nY;
  short _nZ = nZ;
  short _nTimesteps = nTimesteps;

#ifdef TIME
  gettimeofday(&start, 0);
#endif

#pragma scop
  /* Kernel of the code */
  for (t = 0; t < _nTimesteps; t++) {
    for (z = 0; z < _nZ; z++) {
      for (y = 0; y < _nY; y++) {
        for (x = 0; x < _nX; x++) {
          lbm_kernel(
              grid[t % 2][z + 0][y + 0][x + 0][C],
              grid[t % 2][z + 0][y + 0][x==0?_nX-1:x-1][E],
              grid[t % 2][z + 0][y + 0][x==_nX-1?0:x+1][W],
              grid[t % 2][z + 0][y==0?_nY-1:y-1][x + 0][N],
              grid[t % 2][z + 0][y==_nY-1?0:y+1][x + 0][S],
              grid[t % 2][z==0?_nZ-1:z-1][y + 0][x + 0][T],
              grid[t % 2][z==_nZ-1?0:z+1][y + 0][x + 0][B],
              grid[t % 2][z + 0][y==0?_nY-1:y-1][x==0?_nX-1:x-1][NE],
              grid[t % 2][z + 0][y==0?_nY-1:y-1][x==_nX-1?0:x+1][NW],
              grid[t % 2][z + 0][y==_nY-1?0:y+1][x==0?_nX-1:x-1][SE],
              grid[t % 2][z + 0][y==_nY-1?0:y+1][x==_nX-1?0:x+1][SW],
              grid[t % 2][z==0?_nZ-1:z-1][y + 0][x==0?_nX-1:x-1][TE],
              grid[t % 2][z==0?_nZ-1:z-1][y + 0][x==_nX-1?0:x+1][TW],
              grid[t % 2][z==_nZ-1?0:z+1][y + 0][x==0?_nX-1:x-1][BE],
              grid[t % 2][z==_nZ-1?0:z+1][y + 0][x==_nX-1?0:x+1][BW],
              grid[t % 2][z==0?_nZ-1:z-1][y==0?_nY-1:y-1][x + 0][TN],
              grid[t % 2][z==0?_nZ-1:z-1][y==_nY-1?0:y+1][x + 0][TS],
              grid[t % 2][z==_nZ-1?0:z+1][y==0?_nY-1:y-1][x + 0][BN],
              grid[t % 2][z==_nZ-1?0:z+1][y==_nY-1?0:y+1][x + 0][BS],
              &grid[(t + 1) % 2][z][y][x][C],
              &grid[(t + 1) % 2][z][y][x][E],
              &grid[(t + 1) % 2][z][y][x][W],
              &grid[(t + 1) % 2][z][y][x][N],
              &grid[(t + 1) % 2][z][y][x][S],
              &grid[(t + 1) % 2][z][y][x][T],
              &grid[(t + 1) % 2][z][y][x][B],
              &grid[(t + 1) % 2][z][y][x][NE],
              &grid[(t + 1) % 2][z][y][x][NW],
              &grid[(t + 1) % 2][z][y][x][SE],
              &grid[(t + 1) % 2][z][y][x][SW],
              &grid[(t + 1) % 2][z][y][x][TE],
              &grid[(t + 1) % 2][z][y][x][TW],
              &grid[(t + 1) % 2][z][y][x][BE],
              &grid[(t + 1) % 2][z][y][x][BW],
              &grid[(t + 1) % 2][z][y][x][TN],
              &grid[(t + 1) % 2][z][y][x][TS],
              &grid[(t + 1) % 2][z][y][x][BN],
              &grid[(t + 1) % 2][z][y][x][BS],
              t, z, y, x);
        }
      }
    }
    /*  #<{(| East Face |)}># */
    /* for (z=0;z<_nZ;z++){ */
    /*     for (y=0;y<_nY;y++){ */
    /*         grid[(t+1)%2][z][y][0][E]=grid[(t+1)%2][z][y][nX][E]; */
    /*         grid[(t+1)%2][z][y][0][NE]=grid[(t+1)%2][z][y][nX][NE]; */
    /*         grid[(t+1)%2][z][y][0][SE]=grid[(t+1)%2][z][y][nX][SE]; */
    /*         grid[(t+1)%2][z][y][0][TE]=grid[(t+1)%2][z][y][nX][TE]; */
    /*         grid[(t+1)%2][z][y][0][BE]=grid[(t+1)%2][z][y][nX][BE]; */
    /*     } */
    /* } */
    /*  */
    /*  #<{(| West Face |)}># */
    /* for (z=0;z<_nZ;z++){ */
    /*     for (y=0;y<_nY;y++){ */
    /*         grid[(t+1)%2][z][y][_nX-1][W]=grid[(t+1)%2][z][y][0][W]; */
    /*         grid[(t+1)%2][z][y][_nX-1][NW]=grid[(t+1)%2][z][y][0][NW]; */
    /*         grid[(t+1)%2][z][y][_nX-1][SW]=grid[(t+1)%2][z][y][0][SW]; */
    /*         grid[(t+1)%2][z][y][_nX-1][TW]=grid[(t+1)%2][z][y][0][TW]; */
    /*         grid[(t+1)%2][z][y][_nX-1][BW]=grid[(t+1)%2][z][y][0][BW]; */
    /*     } */
    /* } */
    /*  */
    /*  #<{(| South Face |)}># */
    /* for (z=0;z<_nZ;z++){ */
    /*     for (x=0;x<_nX;x++){ */
    /*         grid[(t+1)%2][z][_nY-1][x][S]=grid[(t+1)%2][z][0][x][S]; */
    /*         grid[(t+1)%2][z][_nY-1][x][SE]=grid[(t+1)%2][z][0][x][SE]; */
    /*         grid[(t+1)%2][z][_nY-1][x][SW]=grid[(t+1)%2][z][0][x][SW]; */
    /*         grid[(t+1)%2][z][_nY-1][x][TS]=grid[(t+1)%2][z][0][x][TS]; */
    /*         grid[(t+1)%2][z][_nY-1][x][BS]=grid[(t+1)%2][z][0][x][BS]; */
    /*     } */
    /* } */
    /*  #<{(| North Face |)}># */
    /* for (z=0;z<_nZ;z++){ */
    /*     for (x=0;x<_nX;x++){ */
    /*         grid[(t+1)%2][z][0][x][N]=grid[(t+1)%2][z][nY][x][N]; */
    /*         grid[(t+1)%2][z][0][x][NE]=grid[(t+1)%2][z][nY][x][NE]; */
    /*         grid[(t+1)%2][z][0][x][NW]=grid[(t+1)%2][z][nY][x][NW]; */
    /*         grid[(t+1)%2][z][0][x][TN]=grid[(t+1)%2][z][nY][x][TN]; */
    /*         grid[(t+1)%2][z][0][x][BN]=grid[(t+1)%2][z][nY][x][BN]; */
    /*     } */
    /* } */
    /*  #<{(| Top Face |)}># */
    /* for (y=0;y<_nY;y++){ */
    /*     for (x=0;x<_nX;x++){ */
    /*         grid[(t+1)%2][0][y][x][T]=grid[(t+1)%2][nZ][y][x][T]; */
    /*         grid[(t+1)%2][0][y][x][TE]=grid[(t+1)%2][nZ][y][x][TE]; */
    /*         grid[(t+1)%2][0][y][x][TW]=grid[(t+1)%2][nZ][y][x][TW]; */
    /*         grid[(t+1)%2][0][y][x][TN]=grid[(t+1)%2][nZ][y][x][TN]; */
    /*         grid[(t+1)%2][0][y][x][TS]=grid[(t+1)%2][nZ][y][x][TS]; */
    /*     } */
    /* } */
    /*  #<{(| Bottom Face |)}># */
    /* for (y=0;y<_nY;y++){ */
    /*     for (x=0;x<_nX;x++){ */
    /*         grid[(t+1)%2][_nZ-1][y][x][B]=grid[(t+1)%2][0][y][x][B]; */
    /*         grid[(t+1)%2][_nZ-1][y][x][BE]=grid[(t+1)%2][0][y][x][BE]; */
    /*         grid[(t+1)%2][_nZ-1][y][x][BW]=grid[(t+1)%2][0][y][x][BW]; */
    /*         grid[(t+1)%2][_nZ-1][y][x][BN]=grid[(t+1)%2][0][y][x][BN]; */
    /*         grid[(t+1)%2][_nZ-1][y][x][BS]=grid[(t+1)%2][0][y][x][BS]; */
    /*     } */
    /* } */
  }
#pragma endscop

#ifdef TIME
  gettimeofday(&end, 0);

  ts_return = timeval_subtract(&result, &end, &start);
  tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

  printf("\tTime taken : %7.5lfm\n", tdiff / 60.0);
  printf("\tMLUPS      : %7.5lf\n", (total_lattice_pts / (1.0e6 * tdiff)));
#endif

#ifdef DEBUG
  /* Dump rho, uX, uY for the entire domain to verify results */
  dumpVelocities(t);
#endif

  return 0;
}

void lbm_kernel(
    double in_c, double in_e, double in_w, double in_n, double in_s,
    double in_t, double in_b, double in_ne, double in_nw, double in_se,
    double in_sw, double in_te, double in_tw, double in_be, double in_bw,
    double in_tn, double in_ts, double in_bn, double in_bs,
    double *restrict out_c, double *restrict out_e, double *restrict out_w,
    double *restrict out_n, double *restrict out_s, double *restrict out_t,
    double *restrict out_b, double *restrict out_ne, double *restrict out_nw,
    double *restrict out_se, double *restrict out_sw, double *restrict out_te,
    double *restrict out_tw, double *restrict out_be, double *restrict out_bw,
    double *restrict out_tn, double *restrict out_ts, double *restrict out_bn,
    double *restrict out_bs, int t, int z, int y, int x) {
  /* %%INIT%% */
  double lbm_sum_e, lbm_sum_w, lbm_sum_n, lbm_sum_s, lbm_sum_t, lbm_sum_t_p,
      lbm_sum_b, lbm_sum_b_p;
  double lbm_rho, lbm_inv_rho, lbm_omega_rho;
  double lbm_uX, lbm_uY, lbm_uZ;
  double lbm_uX2, lbm_uY2, lbm_uZ2;
  double lbm_u2, lbm_1_minus_u2;
  double lbm_omega_w1_rho, lbm_omega_w2_rho, lbm_omega_w3_rho;
  double lbm_30_uX, lbm_45_uX2, lbm_30_uY, lbm_45_uY2, lbm_30_uZ, lbm_45_uZ2;
  double lbm_uX_plus_uY, lbm_uX_minus_uY, lbm_uY_plus_uZ, lbm_uY_minus_uZ,
      lbm_uX_plus_uZ, lbm_uX_minus_uZ;
  double lbm_30_uX_plus_uY, lbm_30_uX_minus_uY, lbm_45_uX_plus_uY,
      lbm_45_uX_minus_uY, lbm_30_uY_plus_uZ, lbm_30_uY_minus_uZ,
      lbm_30_uX_plus_uZ, lbm_30_uX_minus_uZ, lbm_45_uY_plus_uZ,
      lbm_45_uY_minus_uZ, lbm_45_uX_plus_uZ, lbm_45_uX_minus_uZ;
  double lbm_out_c, lbm_out_e, lbm_out_w, lbm_out_n, lbm_out_s, lbm_out_t,
      lbm_out_b, lbm_out_ne, lbm_out_nw, lbm_out_se, lbm_out_sw, lbm_out_tn,
      lbm_out_ts, lbm_out_te, lbm_out_tw, lbm_out_bn, lbm_out_bs, lbm_out_be,
      lbm_out_bw;

  double rho, uX, uY, uZ, u2;
  /* %%INIT%% */

  
  /* Algebraic simplification */

  lbm_sum_e = in_e + in_se + in_ne + in_te + in_be;
  lbm_sum_w = in_w + in_sw + in_nw + in_tw + in_bw;
  lbm_sum_n = in_n + in_ne + in_nw + in_tn + in_bn;
  lbm_sum_s = in_s + in_se + in_sw + in_ts + in_bs;
  lbm_sum_t_p = in_t + in_tn + in_ts;
  lbm_sum_t = lbm_sum_t_p + in_te + in_tw;
  lbm_sum_b_p = in_b + in_bn + in_bs;
  lbm_sum_b = lbm_sum_b_p + in_be + in_bw;
  /* 24 add */

  /* Compute rho */
  lbm_rho =
      lbm_sum_e + (in_n + in_c + in_s + lbm_sum_t_p + lbm_sum_b_p) + lbm_sum_w;
  lbm_inv_rho = 1.0 / lbm_rho;

  /* Compute velocity along x-axis */
  lbm_uX = lbm_inv_rho * (lbm_sum_e - lbm_sum_w);

  /* Compute velocity along y-axis */
  lbm_uY = lbm_inv_rho * (lbm_sum_n - lbm_sum_s);

  /* Compute velocity along z-axis */
  lbm_uZ = lbm_inv_rho * (lbm_sum_t - lbm_sum_b);

  /* %%BOUNDARY%% */
  /* Impose the constantly moving upper wall */
  if (z == nZ-1)  {
    lbm_uX = uTopX;
    lbm_uY = uTopY;
    lbm_uZ = 0.00;
  }
  /* %%BOUNDARY%% */

  lbm_uX2 = lbm_uX * lbm_uX;
  lbm_uY2 = lbm_uY * lbm_uY;
  lbm_uZ2 = lbm_uZ * lbm_uZ;

  lbm_u2 = 1.5 * (lbm_uX2 + lbm_uY2 + lbm_uZ2);
  lbm_1_minus_u2 = 1.0 - lbm_u2;

  lbm_omega_rho = omega * lbm_rho;

  lbm_omega_w1_rho = w1 * lbm_omega_rho;
  lbm_omega_w2_rho = w2 * lbm_omega_rho;
  lbm_omega_w3_rho = w3 * lbm_omega_rho;

  lbm_30_uX = 3.0 * lbm_uX;
  lbm_45_uX2 = 4.5 * lbm_uX2;
  lbm_30_uY = 3.0 * lbm_uY;
  lbm_45_uY2 = 4.5 * lbm_uY2;
  lbm_30_uZ = 3.0 * lbm_uZ;
  lbm_45_uZ2 = 4.5 * lbm_uZ2;

  /* ToDo:: Add for uZ */
  lbm_uX_plus_uY = lbm_uX + lbm_uY;
  lbm_uX_minus_uY = lbm_uX - lbm_uY;
  lbm_uY_plus_uZ = lbm_uY + lbm_uZ;
  lbm_uY_minus_uZ = lbm_uY - lbm_uZ;
  lbm_uX_plus_uZ = lbm_uX + lbm_uZ;
  lbm_uX_minus_uZ = lbm_uX - lbm_uZ;

  lbm_30_uX_plus_uY = 3.0 * lbm_uX_plus_uY;
  lbm_30_uX_minus_uY = 3.0 * lbm_uX_minus_uY;
  lbm_30_uY_plus_uZ = 3.0 * lbm_uY_plus_uZ;
  lbm_30_uY_minus_uZ = 3.0 * lbm_uY_minus_uZ;
  lbm_30_uX_plus_uZ = 3.0 * lbm_uX_plus_uZ;
  lbm_30_uX_minus_uZ = 3.0 * lbm_uX_minus_uZ;

  lbm_45_uX_plus_uY = 4.5 * lbm_uX_plus_uY * lbm_uX_plus_uY;
  lbm_45_uX_minus_uY = 4.5 * lbm_uX_minus_uY * lbm_uX_minus_uY;
  lbm_45_uY_plus_uZ = 4.5 * lbm_uY_plus_uZ * lbm_uY_plus_uZ;
  lbm_45_uY_minus_uZ = 4.5 * lbm_uY_minus_uZ * lbm_uY_minus_uZ;
  lbm_45_uX_plus_uZ = 4.5 * lbm_uX_plus_uZ * lbm_uX_plus_uZ;
  lbm_45_uX_minus_uZ = 4.5 * lbm_uX_minus_uZ * lbm_uX_minus_uZ;

  /* Compute and keep new PDFs */
  lbm_out_c = one_minus_omega * in_c + lbm_omega_w1_rho * (lbm_1_minus_u2);

  lbm_out_n = one_minus_omega * in_n +
              lbm_omega_w2_rho * (lbm_1_minus_u2 + lbm_30_uY + lbm_45_uY2);
  lbm_out_s = one_minus_omega * in_s +
              lbm_omega_w2_rho * (lbm_1_minus_u2 - lbm_30_uY + lbm_45_uY2);
  lbm_out_e = one_minus_omega * in_e +
              lbm_omega_w2_rho * (lbm_1_minus_u2 + lbm_30_uX + lbm_45_uX2);
  lbm_out_w = one_minus_omega * in_w +
              lbm_omega_w2_rho * (lbm_1_minus_u2 - lbm_30_uX + lbm_45_uX2);
  lbm_out_t = one_minus_omega * in_t +
              lbm_omega_w2_rho * (lbm_1_minus_u2 + lbm_30_uZ + lbm_45_uZ2);
  lbm_out_b = one_minus_omega * in_b +
              lbm_omega_w2_rho * (lbm_1_minus_u2 - lbm_30_uZ + lbm_45_uZ2);

  lbm_out_ne = one_minus_omega * in_ne +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 + lbm_30_uX_plus_uY + lbm_45_uX_plus_uY);
  lbm_out_nw = one_minus_omega * in_nw +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 - lbm_30_uX_minus_uY + lbm_45_uX_minus_uY);
  lbm_out_se = one_minus_omega * in_se +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 + lbm_30_uX_minus_uY + lbm_45_uX_minus_uY);
  lbm_out_sw = one_minus_omega * in_sw +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 - lbm_30_uX_plus_uY + lbm_45_uX_plus_uY);

  lbm_out_tn = one_minus_omega * in_tn +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 + lbm_30_uY_plus_uZ + lbm_45_uY_plus_uZ);
  lbm_out_ts = one_minus_omega * in_ts +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 - lbm_30_uY_minus_uZ + lbm_45_uY_minus_uZ);
  lbm_out_te = one_minus_omega * in_te +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 + lbm_30_uX_plus_uZ + lbm_45_uX_plus_uZ);
  lbm_out_tw = one_minus_omega * in_tw +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 - lbm_30_uX_minus_uZ + lbm_45_uX_minus_uZ);

  lbm_out_bn = one_minus_omega * in_bn +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 + lbm_30_uY_minus_uZ + lbm_45_uY_minus_uZ);
  lbm_out_bs = one_minus_omega * in_bs +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 - lbm_30_uY_plus_uZ + lbm_45_uY_plus_uZ);
  lbm_out_be = one_minus_omega * in_be +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 + lbm_30_uX_minus_uZ + lbm_45_uX_minus_uZ);
  lbm_out_bw = one_minus_omega * in_bw +
               lbm_omega_w3_rho *
                   (lbm_1_minus_u2 - lbm_30_uX_plus_uZ + lbm_45_uX_plus_uZ);

  *out_c = lbm_out_c;

  *out_n = lbm_out_n;
  *out_s = lbm_out_s;
  *out_e = lbm_out_e;
  *out_w = lbm_out_w;
  *out_t = lbm_out_t;
  *out_b = lbm_out_b;

  *out_ne = lbm_out_ne;
  *out_nw = lbm_out_nw;
  *out_se = lbm_out_se;
  *out_sw = lbm_out_sw;

  *out_tn = lbm_out_tn;
  *out_ts = lbm_out_ts;
  *out_te = lbm_out_te;
  *out_tw = lbm_out_tw;

  *out_bn = lbm_out_bn;
  *out_bs = lbm_out_bs;
  *out_be = lbm_out_be;
  *out_bw = lbm_out_bw;

  /* Algebraic simplification */

  /* #<{(| Compute rho |)}># */
  /* rho = in_n + in_s + in_e + in_w + in_t + in_b + in_ne + in_nw + in_se + */
  /*       in_sw + in_tn + in_ts + in_te + in_tw + in_bn + in_bs + in_be + in_bw
   * + */
  /*       in_c; */

  /* #<{(| Compute velocity along x-axis |)}># */
  /* uX = (in_se + in_ne + in_te + in_be + in_e) - */
  /*      (in_sw + in_nw + in_tw + in_bw + in_w); */
  /* uX = uX / rho; */

  /* #<{(| Compute velocity along y-axis |)}># */
  /* uY = (in_nw + in_ne + in_tn + in_bn + in_n) - */
  /*      (in_sw + in_se + in_ts + in_bs + in_s); */
  /* uY = uY / rho; */

  /* #<{(| Compute velocity along z-axis |)}># */
  /* uZ = (in_tn + in_ts + in_te + in_tw + in_t) - */
  /*      (in_bn + in_bs + in_be + in_bw + in_b); */
  /* uZ = uZ / rho; */

  /* #<{(| %%BOUNDARY%% |)}># */
  /* #<{(| Impose the constantly moving upper wall |)}># */
  /* if (((z == nZ + 2) && (x > 1 && x < nX + 2) && (y > 1 && y < nY + 2))) { */
  /*   uX = uTopX; */
  /*   uY = uTopY; */
  /*   uZ = 0.00; */
  /* } */
  /* #<{(| %%BOUNDARY%% |)}># */

  /* u2 = 1.5 * (uX * uX + uY * uY + uZ * uZ); */

  /* #<{(| Compute and keep new PDFs |)}># */
  /* *out_c = (1.0 - omega) * in_c + (omega * (w1 * rho * (1.0 - u2))); */

  /* *out_n  = (1.0 - omega) * in_n + (omega * (w2 * rho * (1.0 + 3.0 * uY + 4.5
   * * uY * uY - u2))); */
  /* *out_s  = (1.0 - omega) * in_s + (omega * (w2 * rho * (1.0 - 3.0 * uY + 4.5
   * * uY * uY - u2))); */
  /* *out_e  = (1.0 - omega) * in_e + (omega * (w2 * rho * (1.0 + 3.0 * uX + 4.5
   * * uX * uX - u2))); */
  /* *out_w  = (1.0 - omega) * in_w + (omega * (w2 * rho * (1.0 - 3.0 * uX + 4.5
   * * uX * uX - u2))); */
  /* *out_t  = (1.0 - omega) * in_t + (omega * (w2 * rho * (1.0 + 3.0 * uZ + 4.5
   * * uZ * uZ - u2))); */
  /* *out_b  = (1.0 - omega) * in_b + (omega * (w2 * rho * (1.0 - 3.0 * uZ + 4.5
   * * uZ * uZ - u2))); */

  /* *out_ne = (1.0 - omega) * in_ne + (omega * (w3 * rho * (1.0 + 3.0 * (uX +
   * uY) + 4.5 * (uX + uY) * (uX + uY) - u2))); */
  /* *out_nw = (1.0 - omega) * in_nw + (omega * (w3 * rho * (1.0 + 3.0 * (-uX +
   * uY) + 4.5 * (-uX + uY) * (-uX + uY) - u2))); */
  /* *out_se = (1.0 - omega) * in_se + (omega * (w3 * rho * (1.0 + 3.0 * (uX -
   * uY) + 4.5 * (uX - uY) * (uX - uY) - u2))); */
  /* *out_sw = (1.0 - omega) * in_sw + (omega * (w3 * rho * (1.0 + 3.0 * (-uX -
   * uY) + 4.5 * (-uX - uY) * (-uX - uY) - u2))); */

  /* *out_tn = (1.0 - omega) * in_tn + (omega * (w3 * rho * (1.0 + 3.0 * (+uY +
   * uZ) + 4.5 * (+uY + uZ) * (+uY + uZ) - u2))); */
  /* *out_ts = (1.0 - omega) * in_ts + (omega * (w3 * rho * (1.0 + 3.0 * (-uY +
   * uZ) + 4.5 * (-uY + uZ) * (-uY + uZ) - u2))); */
  /* *out_te = (1.0 - omega) * in_te + (omega * (w3 * rho * (1.0 + 3.0 * (+uX +
   * uZ) + 4.5 * (+uX + uZ) * (+uX + uZ) - u2))); */
  /* *out_tw = (1.0 - omega) * in_tw + (omega * (w3 * rho * (1.0 + 3.0 * (-uX +
   * uZ) + 4.5 * (-uX + uZ) * (-uX + uZ) - u2))); */

  /* *out_bn = (1.0 - omega) * in_bn + (omega * (w3 * rho * (1.0 + 3.0 * (+uY -
   * uZ) + 4.5 * (+uY - uZ) * (+uY - uZ) - u2))); */
  /* *out_bs = (1.0 - omega) * in_bs + (omega * (w3 * rho * (1.0 + 3.0 * (-uY -
   * uZ) + 4.5 * (-uY - uZ) * (-uY - uZ) - u2))); */
  /* *out_be = (1.0 - omega) * in_be + (omega * (w3 * rho * (1.0 + 3.0 * (+uX -
   * uZ) + 4.5 * (+uX - uZ) * (+uX - uZ) - u2))); */
  /* *out_bw = (1.0 - omega) * in_bw + (omega * (w3 * rho * (1.0 + 3.0 * (-uX -
   * uZ) + 4.5 * (-uX - uZ) * (-uX - uZ) - u2))); */
}

void dumpVelocities(int t) {
  int z, y, x;
  double rho, uX, uY, uZ;

  for (z = 1; z < nZ - 1; z++) {
    for (y = 1; y < nY - 1; y++) {
      for (x = 1; x < nX - 1; x++) {
        /* Compute rho */
        rho = grid[t % 2][z + 0][y - 1][x + 0][N] +
              grid[t % 2][z + 0][y + 1][x + 0][S] +
              grid[t % 2][z + 0][y + 0][x - 1][E] +
              grid[t % 2][z + 0][y + 0][x + 1][W] +
              grid[t % 2][z - 1][y + 0][x + 0][T] +
              grid[t % 2][z + 1][y + 0][x + 0][B] +
              grid[t % 2][z + 0][y - 1][x - 1][NE] +
              grid[t % 2][z + 0][y - 1][x + 1][NW] +
              grid[t % 2][z + 0][y + 1][x - 1][SE] +
              grid[t % 2][z + 0][y + 1][x + 1][SW] +
              grid[t % 2][z - 1][y - 1][x + 0][TN] +
              grid[t % 2][z - 1][y + 1][x + 0][TS] +
              grid[t % 2][z - 1][y + 0][x - 1][TE] +
              grid[t % 2][z - 1][y + 0][x + 1][TW] +
              grid[t % 2][z + 1][y - 1][x + 0][BN] +
              grid[t % 2][z + 1][y + 1][x + 0][BS] +
              grid[t % 2][z + 1][y + 0][x - 1][BE] +
              grid[t % 2][z + 1][y + 0][x + 1][BW] +
              grid[t % 2][z + 0][y + 0][x + 0][C];

        /* Compute velocity along x-axis */
        uX = (grid[t % 2][z + 0][y + 1][x - 1][SE] +
              grid[t % 2][z + 0][y - 1][x - 1][NE] +
              grid[t % 2][z - 1][y + 0][x - 1][TE] +
              grid[t % 2][z + 1][y + 0][x - 1][BE] +
              grid[t % 2][z + 0][y + 0][x - 1][E]) -
             (grid[t % 2][z + 0][y + 1][x + 1][SW] +
              grid[t % 2][z + 0][y - 1][x + 1][NW] +
              grid[t % 2][z - 1][y + 0][x + 1][TW] +
              grid[t % 2][z + 1][y + 0][x + 1][BW] +
              grid[t % 2][z + 0][y + 0][x + 1][W]);
        uX = uX / rho;

        /* Compute velocity along y-axis */
        uY = (grid[t % 2][z + 0][y - 1][x + 1][NW] +
              grid[t % 2][z + 0][y - 1][x - 1][NE] +
              grid[t % 2][z - 1][y - 1][x + 0][TN] +
              grid[t % 2][z + 1][y - 1][x + 0][BN] +
              grid[t % 2][z + 0][y - 1][x + 0][N]) -
             (grid[t % 2][z + 0][y + 1][x + 1][SW] +
              grid[t % 2][z + 0][y + 1][x - 1][SE] +
              grid[t % 2][z - 1][y + 1][x + 0][TS] +
              grid[t % 2][z + 1][y + 1][x + 0][BS] +
              grid[t % 2][z + 0][y + 1][x + 0][S]);
        uY = uY / rho;

        /* Compute velocity along z-axis */
        uZ = (grid[t % 2][z - 1][y - 1][x + 0][TN] +
              grid[t % 2][z - 1][y + 1][x + 0][TS] +
              grid[t % 2][z - 1][y + 0][x - 1][TE] +
              grid[t % 2][z - 1][y + 0][x + 1][TW] +
              grid[t % 2][z - 1][y + 0][x + 0][T]) -
             (grid[t % 2][z + 1][y - 1][x + 0][BN] +
              grid[t % 2][z + 1][y + 1][x + 0][BS] +
              grid[t % 2][z + 1][y + 0][x - 1][BE] +
              grid[t % 2][z + 1][y + 0][x + 1][BW] +
              grid[t % 2][z + 1][y + 0][x + 0][B]);
        uZ = uZ / rho;

        fprintf(stderr, "%0.6lf,%0.6lf,%0.6lf,%0.6lf\n", rho, uX, uY, uZ);
      }
    }
  }
}

int timeval_subtract(struct timeval *result, struct timeval *x,
                     struct timeval *y) {
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;

    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }

  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;

    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  return x->tv_sec < y->tv_sec;
}

/* icc -O3 -fp-model precise -restrict -DDEBUG -DTIME ldc_d3q19.c -o
 * ldc_d3q19.elf */
