#include <omp.h>
#include <math.h>
#define ceild(n, d) ceil(((double)(n)) / ((double)(d)))
#define floord(n, d) floor(((double)(n)) / ((double)(d)))
#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

/*************************************************************************/
/* Lid Driven Cavity(LDC) - 3D simulation using LBM with a D3Q27 lattice */
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
#define nK 27  // D3Q27
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
#define TNE 19
#define TNW 20
#define TSE 21
#define TSW 22
#define BNE 23
#define BNW 24
#define BSE 25
#define BSW 26

/* Tunable parameters */
const double re = 100;
const double uTopX = 0.05;
const double uTopY = 0.02;

/* Calculated parameters: Set in main */
double omega = 0.0;

const double w1 = 8.0 / 27.0;
const double w2 = 2.0 / 27.0;
const double w3 = 1.0 / 54.0;
const double w4 = 1.0 / 216.0;

double grid[2][nZ + 2 + 4][nY + 2 + 2][nX + 2 + 2][nK];

void lbm_kernel(double, double, double, double, double, double, double, double,
                double, double, double, double, double, double, double, double,
                double, double, double, double, double, double, double, double,
                double, double, double, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, int, int, int, int);

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

  printf(
      "3D Lid Driven Cavity simulation with D3Q27 lattice:\n"
      "\tscheme     : 3-Grid, Fused, Pull\n"
      "\tgrid size  : %d x %d x %d = %.2lf * 10^3 Cells\n"
      "\tnTimeSteps : %d\n"
      "\tRe         : %.2lf\n"
      "\tuTopX      : %.6lf\n"
      "\tuTopY      : %.6lf\n"
      "\tomega      : %.6lf\n",
      nX, nY, nZ, nX * nY * nZ / 1.0e3, nTimesteps, re, uTopX, uTopY, omega);

  /* Initialize all 27 PDFs for each point in the domain to w1, w2 or w3
   * accordingly */
  for (z = 0; z < nZ + 2 + 4; z++) {
    for (y = 0; y < nY + 2 + 2; y++) {
      for (x = 0; x < nX + 2 + 2; x++) {
        grid[0][z][y][x][0] = w1;
        grid[1][z][y][x][0] = w1;

        for (k = 1; k < 7; k++) {
          grid[0][z][y][x][k] = w2;
          grid[1][z][y][x][k] = w2;
        }

        for (k = 7; k < 19; k++) {
          grid[0][z][y][x][k] = w3;
          grid[1][z][y][x][k] = w3;
        }

        for (k = 19; k < nK; k++) {
          grid[0][z][y][x][k] = w4;
          grid[1][z][y][x][k] = w4;
        }
      }
    }
  }

  /* To satisfy PET */
  short _nX = nX + 3;
  short _nY = nY + 3;
  short _nZ = nZ + 4;
  short _nTimesteps = nTimesteps;

#ifdef TIME
  gettimeofday(&start, 0);
#endif

  int t1, t2, t3, t4, t5, t6, t7, t8;
  int lb, ub, lbp, ubp, lb2, ub2;
  register int lbv, ubv;
  /* Start of CLooG code */
  if ((_nTimesteps >= 1) && (_nX >= 2) && (_nY >= 2) && (_nZ >= 3)) {
    for (t1 = -1; t1 <= floord(3 * _nTimesteps + _nZ - 4, 16); t1++) {
      lbp = max(max(ceild(2 * t1, 3), ceild(8 * t1 - _nTimesteps + 1, 4)),
                ceild(16 * t1 - _nTimesteps + 3, 16));
      ubp = min(
          min(floord(_nTimesteps + _nZ - 2, 8), floord(8 * t1 + _nZ + 6, 12)),
          2 * t1 + 2);
#pragma omp parallel for private(lbv, ubv, t3, t4, t5, t6, t7, t8)
      for (t2 = lbp; t2 <= ubp; t2++) {
        for (t3 = max(max(0, ceild(2 * t1 - t2 - 1, 2)),
                      ceild(8 * t2 - _nZ - 5, 8));
             t3 <= min(min(min(floord(_nTimesteps + _nY - 2, 8),
                               floord(8 * t2 + _nY + 4, 8)),
                           floord(8 * t1 - 4 * t2 + _nY + 10, 8)),
                       floord(16 * t1 - 16 * t2 + _nZ + _nY + 13, 8));
             t3++) {
          for (t4 = max(max(max(0, ceild(2 * t1 - t2 - 1, 2)),
                            ceild(8 * t2 - _nZ - 5, 8)),
                        ceild(8 * t3 - _nY - 5, 8));
               t4 <= min(min(min(min(floord(_nTimesteps + _nX - 2, 8),
                                     floord(8 * t2 + _nX + 4, 8)),
                                 floord(8 * t3 + _nX + 5, 8)),
                             floord(8 * t1 - 4 * t2 + _nX + 10, 8)),
                         floord(16 * t1 - 16 * t2 + _nZ + _nX + 13, 8));
               t4++) {
            for (t5 = max(max(max(max(max(0, 8 * t1 - 4 * t2),
                                      16 * t1 - 16 * t2 + 2),
                                  8 * t2 - _nZ + 1),
                              8 * t3 - _nY + 1),
                          8 * t4 - _nX + 1);
                 t5 <= min(min(min(min(min(_nTimesteps - 1, 8 * t2 + 5),
                                       8 * t3 + 6),
                                   8 * t4 + 6),
                               8 * t1 - 4 * t2 + 11),
                           16 * t1 - 16 * t2 + _nZ + 14);
                 t5++) {

              /* Hoisted loop conditional */
              if (t5 % 2 == 0) {
                for (t6 = max(max(8 * t2, t5 + 2),
                              -16 * t1 + 16 * t2 + 2 * t5 - 15);
                     t6 <= min(min(8 * t2 + 7, -16 * t1 + 16 * t2 + 2 * t5),
                               t5 + _nZ - 1);
                     t6++) {
                  for (t7 = max(8 * t3, t5 + 1);
                       t7 <= min(8 * t3 + 7, t5 + _nY - 1); t7++) {
                    lbv = max(8 * t4, t5 + 1);
                    ubv = min(8 * t4 + 7, t5 + _nX - 1);

#pragma ivdep
#pragma vector always
                    for (t8 = lbv; t8 <= ubv; t8++) {
                      lbm_kernel(
                          grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][0],
                          grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8) - 1][1],
                          grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8) + 1][2],
                          grid[0][(-t5 + t6)][(-t5 + t7) - 1][(-t5 + t8)][3],
                          grid[0][(-t5 + t6)][(-t5 + t7) + 1][(-t5 + t8)][4],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7)][(-t5 + t8)][5],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7)][(-t5 + t8)][6],
                          grid[0][(-t5 + t6)][(-t5 + t7) - 1][(-t5 + t8) - 1]
                              [7],
                          grid[0][(-t5 + t6)][(-t5 + t7) - 1][(-t5 + t8) + 1]
                              [8],
                          grid[0][(-t5 + t6)][(-t5 + t7) + 1][(-t5 + t8) - 1]
                              [9],
                          grid[0][(-t5 + t6)][(-t5 + t7) + 1][(-t5 + t8) + 1]
                              [10],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7)][(-t5 + t8) - 1]
                              [11],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7)][(-t5 + t8) + 1]
                              [12],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7)][(-t5 + t8) - 1]
                              [13],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7)][(-t5 + t8) + 1]
                              [14],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7) - 1][(-t5 + t8)]
                              [15],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7) + 1][(-t5 + t8)]
                              [16],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7) - 1][(-t5 + t8)]
                              [17],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7) + 1][(-t5 + t8)]
                              [18],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7) - 1]
                              [(-t5 + t8) - 1][19],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7) - 1]
                              [(-t5 + t8) + 1][20],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7) + 1]
                              [(-t5 + t8) - 1][21],
                          grid[0][(-t5 + t6) - 1][(-t5 + t7) + 1]
                              [(-t5 + t8) + 1][22],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7) - 1]
                              [(-t5 + t8) - 1][23],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7) - 1]
                              [(-t5 + t8) + 1][24],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7) + 1]
                              [(-t5 + t8) - 1][25],
                          grid[0][(-t5 + t6) + 1][(-t5 + t7) + 1]
                              [(-t5 + t8) + 1][26],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][0],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][1],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][2],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][3],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][4],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][5],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][6],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][7],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][8],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][9],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][10],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][11],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][12],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][13],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][14],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][15],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][16],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][17],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][18],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][19],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][20],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][21],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][22],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][23],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][24],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][25],
                          &grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][26],
                          (t5), ((-t5 + t6)), ((-t5 + t7)), ((-t5 + t8)));
                      ;
                    }
                  }
                }
              } else {
                for (t6 = max(max(8 * t2, t5 + 2),
                              -16 * t1 + 16 * t2 + 2 * t5 - 15);
                     t6 <= min(min(8 * t2 + 7, -16 * t1 + 16 * t2 + 2 * t5),
                               t5 + _nZ - 1);
                     t6++) {
                  for (t7 = max(8 * t3, t5 + 1);
                       t7 <= min(8 * t3 + 7, t5 + _nY - 1); t7++) {
                    lbv = max(8 * t4, t5 + 1);
                    ubv = min(8 * t4 + 7, t5 + _nX - 1);

#pragma ivdep
#pragma vector always
                    for (t8 = lbv; t8 <= ubv; t8++) {
                      lbm_kernel(
                          grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][0],
                          grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8) - 1][1],
                          grid[1][(-t5 + t6)][(-t5 + t7)][(-t5 + t8) + 1][2],
                          grid[1][(-t5 + t6)][(-t5 + t7) - 1][(-t5 + t8)][3],
                          grid[1][(-t5 + t6)][(-t5 + t7) + 1][(-t5 + t8)][4],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7)][(-t5 + t8)][5],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7)][(-t5 + t8)][6],
                          grid[1][(-t5 + t6)][(-t5 + t7) - 1][(-t5 + t8) - 1]
                              [7],
                          grid[1][(-t5 + t6)][(-t5 + t7) - 1][(-t5 + t8) + 1]
                              [8],
                          grid[1][(-t5 + t6)][(-t5 + t7) + 1][(-t5 + t8) - 1]
                              [9],
                          grid[1][(-t5 + t6)][(-t5 + t7) + 1][(-t5 + t8) + 1]
                              [10],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7)][(-t5 + t8) - 1]
                              [11],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7)][(-t5 + t8) + 1]
                              [12],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7)][(-t5 + t8) - 1]
                              [13],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7)][(-t5 + t8) + 1]
                              [14],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7) - 1][(-t5 + t8)]
                              [15],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7) + 1][(-t5 + t8)]
                              [16],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7) - 1][(-t5 + t8)]
                              [17],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7) + 1][(-t5 + t8)]
                              [18],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7) - 1]
                              [(-t5 + t8) - 1][19],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7) - 1]
                              [(-t5 + t8) + 1][20],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7) + 1]
                              [(-t5 + t8) - 1][21],
                          grid[1][(-t5 + t6) - 1][(-t5 + t7) + 1]
                              [(-t5 + t8) + 1][22],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7) - 1]
                              [(-t5 + t8) - 1][23],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7) - 1]
                              [(-t5 + t8) + 1][24],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7) + 1]
                              [(-t5 + t8) - 1][25],
                          grid[1][(-t5 + t6) + 1][(-t5 + t7) + 1]
                              [(-t5 + t8) + 1][26],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][0],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][1],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][2],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][3],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][4],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][5],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][6],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][7],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][8],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][9],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][10],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][11],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][12],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][13],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][14],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][15],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][16],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][17],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][18],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][19],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][20],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][21],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][22],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][23],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][24],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][25],
                          &grid[0][(-t5 + t6)][(-t5 + t7)][(-t5 + t8)][26],
                          (t5), ((-t5 + t6)), ((-t5 + t7)), ((-t5 + t8)));
                      ;
                    }
                  }
                }
              }
              /* end hoisted if */
            }
          }
        }
      }
    }
  }
/* End of CLooG code */

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
    double in_tn, double in_ts, double in_bn, double in_bs, double in_tne,
    double in_tnw, double in_tse, double in_tsw, double in_bne, double in_bnw,
    double in_bse, double in_bsw, double *restrict out_c,
    double *restrict out_e, double *restrict out_w, double *restrict out_n,
    double *restrict out_s, double *restrict out_t, double *restrict out_b,
    double *restrict out_ne, double *restrict out_nw, double *restrict out_se,
    double *restrict out_sw, double *restrict out_te, double *restrict out_tw,
    double *restrict out_be, double *restrict out_bw, double *restrict out_tn,
    double *restrict out_ts, double *restrict out_bn, double *restrict out_bs,
    double *restrict out_tne, double *restrict out_tnw,
    double *restrict out_tse, double *restrict out_tsw,
    double *restrict out_bne, double *restrict out_bnw,
    double *restrict out_bse, double *restrict out_bsw, int t, int z, int y,
    int x) {
  double rho, uX, uY, uZ, u2;

  if (((z == 2 || z == nZ + 3) && (x > 0 && x < nX + 3) &&
       (y > 0 && y < nY + 3)) ||
      ((x == 1 || x == nX + 2) && (y > 0 && y < nY + 3) &&
       (z > 1 && z < nZ + 4)) ||
      ((y == 1 || y == nY + 2) && (x > 0 && x < nX + 3) &&
       (z > 1 && z < nZ + 4))) {
    /* Bounce back boundary condition at walls */
    *out_c = in_c;

    *out_s = in_n;
    *out_n = in_s;
    *out_w = in_e;
    *out_e = in_w;
    *out_b = in_t;
    *out_t = in_b;

    *out_sw = in_ne;
    *out_se = in_nw;
    *out_nw = in_se;
    *out_ne = in_sw;

    *out_bs = in_tn;
    *out_bn = in_ts;
    *out_bw = in_te;
    *out_be = in_tw;
    *out_ts = in_bn;
    *out_tn = in_bs;
    *out_tw = in_be;
    *out_te = in_bw;

    *out_bsw = in_tne;
    *out_bse = in_tnw;
    *out_bnw = in_tse;
    *out_bne = in_tsw;
    *out_tsw = in_bne;
    *out_tse = in_bnw;
    *out_tnw = in_bse;
    *out_tne = in_bsw;

    return;
  }

  /* Compute rho */
  rho = in_n + in_s + in_e + in_w + in_t + in_b + in_ne + in_nw + in_se +
        in_sw + in_tn + in_ts + in_te + in_tw + in_bn + in_bs + in_be + in_bw +
        in_tne + in_tnw + in_tse + in_tsw + in_bne + in_bnw + in_bse + in_bsw +
        in_c;

  /* Compute velocity along x-axis */
  uX = (in_se + in_ne + in_te + in_be + in_tne + in_tse + in_bne + in_bse +
        in_e) -
       (in_sw + in_nw + in_tw + in_bw + in_tnw + in_tsw + in_bnw + in_bsw +
        in_w);
  uX = uX / rho;

  /* Compute velocity along y-axis */
  uY = (in_nw + in_ne + in_tn + in_bn + in_tne + in_tnw + in_bne + in_bnw +
        in_n) -
       (in_sw + in_se + in_ts + in_bs + in_tse + in_tsw + in_bse + in_bsw +
        in_s);
  uY = uY / rho;

  /* Compute velocity along z-axis */
  uZ = (in_tn + in_ts + in_te + in_tw + in_tne + in_tnw + in_tse + in_tsw +
        in_t) -
       (in_bn + in_bs + in_be + in_bw + in_bne + in_bnw + in_bse + in_bsw +
        in_b);
  uZ = uZ / rho;

  /* Impose the constantly moving upper wall */
  if (((z == nZ + 2) && (x > 1 && x < nX + 2) && (y > 1 && y < nY + 2))) {
    uX = uTopX;
    uY = uTopY;
    uZ = 0.00;
  }

  u2 = 1.5 * (uX * uX + uY * uY + uZ * uZ);

  /* Compute and keep new PDFs */
  *out_c = (1.0 - omega) * in_c + (omega * (w1 * rho * (1.0 - u2)));

  *out_n = (1.0 - omega) * in_n +
           (omega * (w2 * rho * (1.0 + 3.0 * uY + 4.5 * uY * uY - u2)));
  *out_s = (1.0 - omega) * in_s +
           (omega * (w2 * rho * (1.0 - 3.0 * uY + 4.5 * uY * uY - u2)));
  *out_e = (1.0 - omega) * in_e +
           (omega * (w2 * rho * (1.0 + 3.0 * uX + 4.5 * uX * uX - u2)));
  *out_w = (1.0 - omega) * in_w +
           (omega * (w2 * rho * (1.0 - 3.0 * uX + 4.5 * uX * uX - u2)));
  *out_t = (1.0 - omega) * in_t +
           (omega * (w2 * rho * (1.0 + 3.0 * uZ + 4.5 * uZ * uZ - u2)));
  *out_b = (1.0 - omega) * in_b +
           (omega * (w2 * rho * (1.0 - 3.0 * uZ + 4.5 * uZ * uZ - u2)));

  *out_ne =
      (1.0 - omega) * in_ne +
      (omega *
       (w3 * rho * (1.0 + 3.0 * (uX + uY) + 4.5 * (uX + uY) * (uX + uY) - u2)));
  *out_nw = (1.0 - omega) * in_nw +
            (omega * (w3 * rho * (1.0 + 3.0 * (-uX + uY) +
                                  4.5 * (-uX + uY) * (-uX + uY) - u2)));
  *out_se =
      (1.0 - omega) * in_se +
      (omega *
       (w3 * rho * (1.0 + 3.0 * (uX - uY) + 4.5 * (uX - uY) * (uX - uY) - u2)));
  *out_sw = (1.0 - omega) * in_sw +
            (omega * (w3 * rho * (1.0 + 3.0 * (-uX - uY) +
                                  4.5 * (-uX - uY) * (-uX - uY) - u2)));

  *out_tn = (1.0 - omega) * in_tn +
            (omega * (w3 * rho * (1.0 + 3.0 * (+uY + uZ) +
                                  4.5 * (+uY + uZ) * (+uY + uZ) - u2)));
  *out_ts = (1.0 - omega) * in_ts +
            (omega * (w3 * rho * (1.0 + 3.0 * (-uY + uZ) +
                                  4.5 * (-uY + uZ) * (-uY + uZ) - u2)));
  *out_te = (1.0 - omega) * in_te +
            (omega * (w3 * rho * (1.0 + 3.0 * (+uX + uZ) +
                                  4.5 * (+uX + uZ) * (+uX + uZ) - u2)));
  *out_tw = (1.0 - omega) * in_tw +
            (omega * (w3 * rho * (1.0 + 3.0 * (-uX + uZ) +
                                  4.5 * (-uX + uZ) * (-uX + uZ) - u2)));

  *out_bn = (1.0 - omega) * in_bn +
            (omega * (w3 * rho * (1.0 + 3.0 * (+uY - uZ) +
                                  4.5 * (+uY - uZ) * (+uY - uZ) - u2)));
  *out_bs = (1.0 - omega) * in_bs +
            (omega * (w3 * rho * (1.0 + 3.0 * (-uY - uZ) +
                                  4.5 * (-uY - uZ) * (-uY - uZ) - u2)));
  *out_be = (1.0 - omega) * in_be +
            (omega * (w3 * rho * (1.0 + 3.0 * (+uX - uZ) +
                                  4.5 * (+uX - uZ) * (+uX - uZ) - u2)));
  *out_bw = (1.0 - omega) * in_bw +
            (omega * (w3 * rho * (1.0 + 3.0 * (-uX - uZ) +
                                  4.5 * (-uX - uZ) * (-uX - uZ) - u2)));

  *out_tne =
      (1.0 - omega) * in_tne +
      (omega * (w4 * rho * (1.0 + 3.0 * (+uX + uY + uZ) +
                            4.5 * (+uX + uY + uZ) * (+uX + uY + uZ) - u2)));
  *out_tnw =
      (1.0 - omega) * in_tnw +
      (omega * (w4 * rho * (1.0 + 3.0 * (-uX + uY + uZ) +
                            4.5 * (-uX + uY + uZ) * (-uX + uY + uZ) - u2)));
  *out_tse =
      (1.0 - omega) * in_tse +
      (omega * (w4 * rho * (1.0 + 3.0 * (+uX - uY + uZ) +
                            4.5 * (+uX - uY + uZ) * (+uX - uY + uZ) - u2)));
  *out_tsw =
      (1.0 - omega) * in_tsw +
      (omega * (w4 * rho * (1.0 + 3.0 * (-uX - uY + uZ) +
                            4.5 * (-uX - uY + uZ) * (-uX - uY + uZ) - u2)));
  *out_bne =
      (1.0 - omega) * in_bne +
      (omega * (w4 * rho * (1.0 + 3.0 * (+uX + uY - uZ) +
                            4.5 * (+uX + uY - uZ) * (+uX + uY - uZ) - u2)));
  *out_bnw =
      (1.0 - omega) * in_bnw +
      (omega * (w4 * rho * (1.0 + 3.0 * (-uX + uY - uZ) +
                            4.5 * (-uX + uY - uZ) * (-uX + uY - uZ) - u2)));
  *out_bse =
      (1.0 - omega) * in_bse +
      (omega * (w4 * rho * (1.0 + 3.0 * (+uX - uY - uZ) +
                            4.5 * (+uX - uY - uZ) * (+uX - uY - uZ) - u2)));
  *out_bsw =
      (1.0 - omega) * in_bsw +
      (omega * (w4 * rho * (1.0 + 3.0 * (-uX - uY - uZ) +
                            4.5 * (-uX - uY - uZ) * (-uX - uY - uZ) - u2)));
}

void dumpVelocities(int t) {
  int z, y, x;
  double rho, uX, uY, uZ;

  for (z = 3; z < nZ + 3; z++) {
    for (y = 2; y < nY + 2; y++) {
      for (x = 2; x < nX + 2; x++) {
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
              grid[t % 2][z - 1][y - 1][x - 1][TNE] +
              grid[t % 2][z - 1][y - 1][x + 1][TNW] +
              grid[t % 2][z - 1][y + 1][x - 1][TSE] +
              grid[t % 2][z - 1][y + 1][x + 1][TSW] +
              grid[t % 2][z + 1][y - 1][x - 1][BNE] +
              grid[t % 2][z + 1][y - 1][x + 1][BNW] +
              grid[t % 2][z + 1][y + 1][x - 1][BSE] +
              grid[t % 2][z + 1][y + 1][x + 1][BSW] +
              grid[t % 2][z + 0][y + 0][x + 0][C];

        /* Compute velocity along x-axis */
        uX = (grid[t % 2][z + 0][y + 1][x - 1][SE] +
              grid[t % 2][z + 0][y - 1][x - 1][NE] +
              grid[t % 2][z - 1][y + 0][x - 1][TE] +
              grid[t % 2][z + 1][y + 0][x - 1][BE] +
              grid[t % 2][z - 1][y - 1][x - 1][TNE] +
              grid[t % 2][z - 1][y + 1][x - 1][TSE] +
              grid[t % 2][z + 1][y - 1][x - 1][BNE] +
              grid[t % 2][z + 1][y + 1][x - 1][BSE] +
              grid[t % 2][z + 0][y + 0][x - 1][E]) -
             (grid[t % 2][z + 0][y + 1][x + 1][SW] +
              grid[t % 2][z + 0][y - 1][x + 1][NW] +
              grid[t % 2][z - 1][y + 0][x + 1][TW] +
              grid[t % 2][z + 1][y + 0][x + 1][BW] +
              grid[t % 2][z - 1][y - 1][x + 1][TNW] +
              grid[t % 2][z - 1][y + 1][x + 1][TSW] +
              grid[t % 2][z + 1][y - 1][x + 1][BNW] +
              grid[t % 2][z + 1][y + 1][x + 1][BSW] +
              grid[t % 2][z + 0][y + 0][x + 1][W]);
        uX = uX / rho;

        /* Compute velocity along y-axis */
        uY = (grid[t % 2][z + 0][y - 1][x + 1][NW] +
              grid[t % 2][z + 0][y - 1][x - 1][NE] +
              grid[t % 2][z - 1][y - 1][x + 0][TN] +
              grid[t % 2][z + 1][y - 1][x + 0][BN] +
              grid[t % 2][z - 1][y - 1][x - 1][TNE] +
              grid[t % 2][z - 1][y - 1][x + 1][TNW] +
              grid[t % 2][z + 1][y - 1][x - 1][BNE] +
              grid[t % 2][z + 1][y - 1][x + 1][BNW] +
              grid[t % 2][z + 0][y - 1][x + 0][N]) -
             (grid[t % 2][z + 0][y + 1][x + 1][SW] +
              grid[t % 2][z + 0][y + 1][x - 1][SE] +
              grid[t % 2][z - 1][y + 1][x + 0][TS] +
              grid[t % 2][z + 1][y + 1][x + 0][BS] +
              grid[t % 2][z - 1][y + 1][x - 1][TSE] +
              grid[t % 2][z - 1][y + 1][x + 1][TSW] +
              grid[t % 2][z + 1][y + 1][x - 1][BSE] +
              grid[t % 2][z + 1][y + 1][x + 1][BSW] +
              grid[t % 2][z + 0][y + 1][x + 0][S]);
        uY = uY / rho;

        /* Compute velocity along z-axis */
        uZ = (grid[t % 2][z - 1][y - 1][x + 0][TN] +
              grid[t % 2][z - 1][y + 1][x + 0][TS] +
              grid[t % 2][z - 1][y + 0][x - 1][TE] +
              grid[t % 2][z - 1][y + 0][x + 1][TW] +
              grid[t % 2][z - 1][y - 1][x - 1][TNE] +
              grid[t % 2][z - 1][y - 1][x + 1][TNW] +
              grid[t % 2][z - 1][y + 1][x - 1][TSE] +
              grid[t % 2][z - 1][y + 1][x + 1][TSW] +
              grid[t % 2][z - 1][y + 0][x + 0][T]) -
             (grid[t % 2][z + 1][y - 1][x + 0][BN] +
              grid[t % 2][z + 1][y + 1][x + 0][BS] +
              grid[t % 2][z + 1][y + 0][x - 1][BE] +
              grid[t % 2][z + 1][y + 0][x + 1][BW] +
              grid[t % 2][z + 1][y - 1][x - 1][BNE] +
              grid[t % 2][z + 1][y - 1][x + 1][BNW] +
              grid[t % 2][z + 1][y + 1][x - 1][BSE] +
              grid[t % 2][z + 1][y + 1][x + 1][BSW] +
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

/* icc -O3 -fp-model precise -restrict -DDEBUG -DTIME ldc_d3q27.c -o
 * ldc_d3q27.elf */
