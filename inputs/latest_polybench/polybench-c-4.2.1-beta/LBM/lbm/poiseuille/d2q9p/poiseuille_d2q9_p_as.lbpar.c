#include <omp.h>
#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*************************************************************************/
/* 2D Poiseuille sheet flow simulation using LBM with a D2Q9 lattice and */
/* Zou-He BC for inlet and outlet and SBB BC for top and bottom walls    */
/*                                                                       */
/* Scheme: 2-Grid, Pull, Compute, Keep                                   */
/*                                                                       */
/* Lattice naming convention                                             */
/*   c6[NW]  c3[N]  c5[NE]                                               */
/*   c2[W]   c0[C]  c1[E]                                                */
/*   c8[SW]  c4[S]  c7[SE]                                               */
/*                                                                       */
/* Author: Irshad Pananilath <pmirshad+code@gmail.com>                   */
/*                                                                       */
/*************************************************************************/

#include <stdio.h>
#include <sys/time.h>

#ifndef nX
#define nX 1024  // Size of domain along x-axis
#endif

#ifndef nY
#define nY 256  // Size of domain along y-axis
#endif

#ifndef nK
#define nK 9  // D2Q9
#endif

#ifndef nTimesteps
#define nTimesteps 40000  // Number of timesteps for the simulation
#endif

#define C 0
#define E 1
#define W 2
#define N 3
#define S 4
#define NE 5
#define NW 6
#define SE 7
#define SW 8

/* Tunable parameters */
const double tau = 0.9;
const double rho_in = 3.0 * 1.03;
const double rho_out = 3.0 * 1.0;

/* Calculated parameters: Set in main */
double omega = 0.0;
double one_minus_omega = 0.0;

const double w1 = 4.0 / 9.0;
const double w2 = 1.0 / 9.0;
const double w3 = 1.0 / 36.0;

double grid[2][nY][nX][nK] __attribute__((aligned(32)));

void lbm_kernel(double, double, double, double, double, double, double, double,
                double, double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict,
                double *restrict, double *restrict, double *restrict, int, int,
                int);
void dumpVelocities(int timestep);
int timeval_subtract(struct timeval *result, struct timeval *x,
                     struct timeval *y);

int main(void) {
    int t, y, x, k;
    double total_lattice_pts = (double)nY * (double)nX * (double)nTimesteps;

    /* For timekeeping */
    int ts_return = -1;
    struct timeval start, end, result;
    double tdiff = 0.0;

    /* Compute values for global parameters */
    omega = 1.0 / tau;
    one_minus_omega = 1.0 - omega;

    double rho_avg = (rho_in + rho_out) / 2.0;

    printf(
        "2D Poiseuille sheet flow simulation with D2Q9 lattice:\n"
        "\tscheme     : 2-Grid, Fused, Pull\n"
        "\tgrid size  : %d x %d = %.2lf * 10^3 Cells\n"
        "\tnTimeSteps : %d\n"
        "\tomega      : %.6lf\n",
        nX, nY, nX * nY / 1.0e3, nTimesteps, omega);

    /* Initialize all 9 PDFs for each point in the domain to 1.0 */
    for (y = 0; y < nY ; y++) {
        for (x = 0; x < nX; x++) {
            grid[0][y][x][0] = w1 * rho_avg;
            grid[1][y][x][0] = w1 * rho_avg;

            for (k = 1; k < 5; k++) {
                grid[0][y][x][k] = w2 * rho_avg;
                grid[1][y][x][k] = w2 * rho_avg;
            }

            for (k = 5; k < nK; k++) {
                grid[0][y][x][k] = w3 * rho_avg;
                grid[1][y][x][k] = w3 * rho_avg;
            }
        }
    }

    /* To satisfy PET */
    short _nX = nX/2;
    short _nY = nY/2;
    int _nTimesteps = nTimesteps;

#ifdef TIME
    gettimeofday(&start, 0);
#endif

    /* Copyright (C) 1991-2015 Free Software Foundation, Inc.
       This file is part of the GNU C Library.

       The GNU C Library is free software; you can redistribute it and/or
       modify it under the terms of the GNU Lesser General Public
       License as published by the Free Software Foundation; either
       version 2.1 of the License, or (at your option) any later version.

       The GNU C Library is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
       Lesser General Public License for more details.

       You should have received a copy of the GNU Lesser General Public
       License along with the GNU C Library; if not, see
       <http://www.gnu.org/licenses/>.  */
    /* This header is separate from features.h so that the compiler can
       include it implicitly at the start of every compilation.  It must
       not itself include <features.h> or any other header that includes
       <features.h> because the implicit include comes before any feature
       test macros that may be defined in a source file before it first
       explicitly includes a system header.  GCC knows the name of this
       header in order to preinclude it.  */
    /* glibc's intent is to support the IEC 559 math functionality, real
       and complex.  If the GCC (4.9 and later) predefined macros
       specifying compiler intent are available, use them to determine
       whether the overall intent is to support these features; otherwise,
       presume an older compiler has intent to support these features and
       define these macros by default.  */
    /* wchar_t uses Unicode 7.0.0.  Version 7.0 of the Unicode Standard is
       synchronized with ISO/IEC 10646:2012, plus Amendments 1 (published
       on April, 2013) and 2 (not yet published as of February, 2015).
       Additionally, it includes the accelerated publication of U+20BD
       RUBLE SIGN.  Therefore Unicode 7.0.0 is between 10646:2012 and
       10646:2014, and so we use the date ISO/IEC 10646:2012 Amd.1 was
       published.  */
    /* We do not support C11 <threads.h>.  */
    int t1, t2, t3, t4, t5, t6, t7;
    int lb, ub, lbp, ubp, lbd, ubd, lb2, ub2;
    register int lbv, ubv;
    /* Start of CLooG code */
    if ((_nTimesteps >= 1) && (_nX >= 1) && (_nY >= 1)) {
        for (t1=-1; t1<=floord(_nTimesteps-1,16); t1++) {
            lbp=ceild(t1,2);
            ubp=min(floord(_nTimesteps+_nY-1,32),floord(16*t1+_nY+15,32));
            #pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7)
            for (t2=lbp; t2<=ubp; t2++) {
                for (t3=max(0,ceild(t1-1,2)); t3<=min(min(floord(_nTimesteps+_nX-1,32),floord(16*t1+_nX+31,32)),floord(32*t1-32*t2+_nY+_nX+31,32)); t3++) {
                    if ((t1 == 2*t2) && (t1 >= ceild(32*t3-_nX+1,16))) {
                        lbv=32*t3;
                        ubv=min(32*t3+31,16*t1+_nX-1);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            if (t1%2 == 0) {
                                lbm_kernel(grid[ 16*t1 % 2][ 0][ (-16*t1+t7)][0], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7)][3], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7)][4], grid[ 16*t1 % 2][ 0][ (-16*t1+t7) == 0 ? 2 * _nX - 1 : (-16*t1+t7) - 1][1], grid[ 16*t1 % 2][ 0][ (-16*t1+t7) + 1 == 2 * _nX ? 0 : (-16*t1+t7) + 1][2], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7) == 0 ? 2 * _nX - 1 : (-16*t1+t7) - 1][5], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7) + 1 == 2 * _nX ? 0 : (-16*t1+t7) + 1][6], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7) == 0 ? 2 * _nX - 1 : (-16*t1+t7) - 1][7], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7) + 1 == 2 * _nX ? 0 : (-16*t1+t7) + 1][8], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][0], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][3], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][4], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][1], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][2], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][5], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][6], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][7], &grid[( 16*t1 + 1) % 2][ 0][ (-16*t1+t7)][8], ( 16*t1), ( 0), ( (-16*t1+t7)));;
                            }
                        }
                        lbv=max(32*t3,16*t1+1);
                        ubv=min(16*t1+_nX,32*t3+31);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            if (t1%2 == 0) {
                                lbm_kernel(grid[ 16*t1 % 2][ 0][ (16*t1-t7+2*_nX)][0], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX)][3], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX)][4], grid[ 16*t1 % 2][ 0][ (16*t1-t7+2*_nX) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX) - 1][1], grid[ 16*t1 % 2][ 0][ (16*t1-t7+2*_nX) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX) + 1][2], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX) - 1][5], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX) + 1][6], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX) - 1][7], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX) + 1][8], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][0], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][3], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][4], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][1], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][2], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][5], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][6], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][7], &grid[( 16*t1 + 1) % 2][ 0][ (16*t1-t7+2*_nX)][8], ( 16*t1), ( 0), ( (16*t1-t7+2*_nX)));;
                            }
                        }
                    }
                    if ((t1 <= floord(32*t2-_nY,16)) && (t2 <= floord(32*t3+_nY+30,32)) && (t2 >= max(ceild(_nY,32),ceild(32*t3+_nY-_nX+1,32)))) {
                        lbv=max(32*t3,32*t2-_nY);
                        ubv=min(32*t3+31,32*t2-_nY+_nX-1);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t2-_nY) % 2][ _nY][ (-32*t2+t7+_nY)][0], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t2+t7+_nY)][3], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t2+t7+_nY)][4], grid[ (32*t2-_nY) % 2][ _nY][ (-32*t2+t7+_nY) == 0 ? 2 * _nX - 1 : (-32*t2+t7+_nY) - 1][1], grid[ (32*t2-_nY) % 2][ _nY][ (-32*t2+t7+_nY) + 1 == 2 * _nX ? 0 : (-32*t2+t7+_nY) + 1][2], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t2+t7+_nY) == 0 ? 2 * _nX - 1 : (-32*t2+t7+_nY) - 1][5], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t2+t7+_nY) + 1 == 2 * _nX ? 0 : (-32*t2+t7+_nY) + 1][6], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t2+t7+_nY) == 0 ? 2 * _nX - 1 : (-32*t2+t7+_nY) - 1][7], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t2+t7+_nY) + 1 == 2 * _nX ? 0 : (-32*t2+t7+_nY) + 1][8], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][0], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][3], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][4], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][1], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][2], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][5], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][6], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][7], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][8], ( (32*t2-_nY)), ( _nY), ( (-32*t2+t7+_nY)));;
                        }
                        lbv=max(32*t3,32*t2-_nY+1);
                        ubv=min(32*t3+31,32*t2-_nY+_nX);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t2-_nY) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][0], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t2-t7-_nY+2*_nX)][3], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t2-t7-_nY+2*_nX)][4], grid[ (32*t2-_nY) % 2][ _nY][ (32*t2-t7-_nY+2*_nX) == 0 ? 2 * _nX - 1 : (32*t2-t7-_nY+2*_nX) - 1][1], grid[ (32*t2-_nY) % 2][ _nY][ (32*t2-t7-_nY+2*_nX) + 1 == 2 * _nX ? 0 : (32*t2-t7-_nY+2*_nX) + 1][2], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t2-t7-_nY+2*_nX) == 0 ? 2 * _nX - 1 : (32*t2-t7-_nY+2*_nX) - 1][5], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t2-t7-_nY+2*_nX) + 1 == 2 * _nX ? 0 : (32*t2-t7-_nY+2*_nX) + 1][6], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t2-t7-_nY+2*_nX) == 0 ? 2 * _nX - 1 : (32*t2-t7-_nY+2*_nX) - 1][7], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t2-t7-_nY+2*_nX) + 1 == 2 * _nX ? 0 : (32*t2-t7-_nY+2*_nX) + 1][8], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][0], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][3], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][4], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][1], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][2], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][5], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][6], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][7], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][8], ( (32*t2-_nY)), ( _nY), ( (32*t2-t7-_nY+2*_nX)));;
                        }
                    }
                    if ((t1 == 2*t3+1) && (16*t1 == 32*t2-_nY-15)) {
                        if ((16*t1+31*_nY+17)%32 == 0) {
                            if ((_nY+31)%32 == 0) {
                                lbm_kernel(grid[ (16*t1+15) % 2][ _nY][ 0][0], grid[ (16*t1+15) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0][3], grid[ (16*t1+15) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0][4], grid[ (16*t1+15) % 2][ _nY][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (16*t1+15) % 2][ _nY][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][2], grid[ (16*t1+15) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (16*t1+15) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (16*t1+15) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][7], grid[ (16*t1+15) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][8], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][0], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][3], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][4], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][1], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][2], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][5], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][6], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][7], &grid[( (16*t1+15) + 1) % 2][ _nY][ 0][8], ( (16*t1+15)), ( _nY), ( 0));;
                            }
                        }
                    }
                    if ((t1 <= min(floord(32*t3-_nX,16),floord(32*t2+32*t3-_nX-1,32))) && (t1 >= ceild(32*t2+32*t3-_nY-_nX-30,32)) && (t2 <= floord(32*t3+_nY-_nX-1,32)) && (t2 >= ceild(32*t3-_nX-30,32)) && (t3 >= ceild(_nX,32))) {
                        for (t6=max(max(32*t2,32*t3-_nX),-32*t1+32*t2+64*t3-2*_nX-31); t6<=min(min(32*t2+31,-32*t1+32*t2+64*t3-2*_nX),32*t3+_nY-_nX-1); t6++) {
                            lbm_kernel(grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX)][ _nX][0], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) == 0 ? 2 * _nY - 1 : (-32*t3+t6+_nX) - 1][ _nX][3], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) + 1 == 2 * _nY ? 0 : (-32*t3+t6+_nX) + 1][ _nX][4], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX)][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX)][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][2], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) == 0 ? 2 * _nY - 1 : (-32*t3+t6+_nX) - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) == 0 ? 2 * _nY - 1 : (-32*t3+t6+_nX) - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) + 1 == 2 * _nY ? 0 : (-32*t3+t6+_nX) + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][7], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) + 1 == 2 * _nY ? 0 : (-32*t3+t6+_nX) + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][8], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][0], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][3], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][4], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][1], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][2], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][5], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][6], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][7], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][8], ( (32*t3-_nX)), ( (-32*t3+t6+_nX)), ( _nX));;
                        }
                        for (t6=max(max(32*t2,32*t3-_nX+1),-32*t1+32*t2+64*t3-2*_nX-31); t6<=min(min(32*t2+31,32*t3+_nY-_nX),-32*t1+32*t2+64*t3-2*_nX); t6++) {
                            lbm_kernel(grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][0], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY-_nX) - 1][ _nX][3], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY-_nX) + 1][ _nX][4], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][2], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY-_nX) - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY-_nX) - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY-_nX) + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][7], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY-_nX) + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][8], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][0], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][3], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][4], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][1], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][2], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][5], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][6], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][7], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][8], ( (32*t3-_nX)), ( (32*t3-t6+2*_nY-_nX)), ( _nX));;
                        }
                    }
                    if ((t1 == 2*t2) && (16*t1 == 32*t3-_nX-31)) {
                        if (t1%2 == 0) {
                            if ((16*t1+_nX+31)%32 == 0) {
                                lbm_kernel(grid[ (16*t1+31) % 2][ 0][ _nX][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX][3], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX][4], grid[ (16*t1+31) % 2][ 0][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (16*t1+31) % 2][ 0][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][2], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][7], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][8], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][0], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][3], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][4], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][1], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][2], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][5], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][6], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][7], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][8], ( (16*t1+31)), ( 0), ( _nX));;
                            }
                        }
                    }
                    if ((t1 == 2*t2) && (16*t1 == 32*t3-_nX)) {
                        if (t1%2 == 0) {
                            if ((16*t1+_nX)%32 == 0) {
                                lbm_kernel(grid[ 16*t1 % 2][ 0][ _nX][0], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX][3], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX][4], grid[ 16*t1 % 2][ 0][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ 16*t1 % 2][ 0][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][2], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ 16*t1 % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][7], grid[ 16*t1 % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][8], &grid[( 16*t1 + 1) % 2][ 0][ _nX][0], &grid[( 16*t1 + 1) % 2][ 0][ _nX][3], &grid[( 16*t1 + 1) % 2][ 0][ _nX][4], &grid[( 16*t1 + 1) % 2][ 0][ _nX][1], &grid[( 16*t1 + 1) % 2][ 0][ _nX][2], &grid[( 16*t1 + 1) % 2][ 0][ _nX][5], &grid[( 16*t1 + 1) % 2][ 0][ _nX][6], &grid[( 16*t1 + 1) % 2][ 0][ _nX][7], &grid[( 16*t1 + 1) % 2][ 0][ _nX][8], ( 16*t1), ( 0), ( _nX));;
                            }
                        }
                    }
                    if (32*t1 == 32*t2+32*t3-_nY-_nX-31) {
                        if ((_nY+_nX+31)%32 == 0) {
                            lbm_kernel(grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ _nX][0], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX][3], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX][4], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][2], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][7], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][8], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][0], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][3], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][4], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][1], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][2], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][5], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][6], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][7], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ _nX][8], ( (32*t1-32*t2+_nY+31)), ( _nY), ( _nX));;
                        }
                    }
                    if ((t1 <= floord(32*t2-_nY,16)) && (t2 >= ceild(_nY,32)) && (32*t2 == 32*t3+_nY-_nX)) {
                        if ((31*_nY+_nX)%32 == 0) {
                            lbm_kernel(grid[ (32*t2-_nY) % 2][ _nY][ _nX][0], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX][3], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX][4], grid[ (32*t2-_nY) % 2][ _nY][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (32*t2-_nY) % 2][ _nY][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][2], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][7], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][8], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][0], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][3], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][4], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][1], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][2], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][5], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][6], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][7], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][8], ( (32*t2-_nY)), ( _nY), ( _nX));;
                        }
                    }
                    for (t4=max(max(max(max(0,16*t1),32*t1-32*t2+1),32*t2-_nY+1),32*t3-_nX+1); t4<=min(min(min(min(_nTimesteps-1,16*t1+31),32*t2+30),32*t3+30),32*t1-32*t2+_nY+30); t4++) {
                        for (t6=max(max(32*t2,t4),-32*t1+32*t2+2*t4-31); t6<=min(min(32*t2+31,-32*t1+32*t2+2*t4),t4+_nY-1); t6++) {
                            lbv=max(32*t3,t4);
                            ubv=min(32*t3+31,t4+_nX-1);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (-t4+t6)][ (-t4+t7)][0], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (-t4+t7)][3], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (-t4+t7)][4], grid[ t4 % 2][ (-t4+t6)][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][1], grid[ t4 % 2][ (-t4+t6)][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][2], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][5], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][6], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][7], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][8], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][0], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][3], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][4], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][1], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][2], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][5], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][6], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][7], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][8], ( t4), ( (-t4+t6)), ( (-t4+t7)));;
                            }
                        }
                        for (t6=max(max(32*t2,t4+1),-32*t1+32*t2+2*t4-31); t6<=min(min(32*t2+31,t4+_nY),-32*t1+32*t2+2*t4); t6++) {
                            lbv=max(32*t3,t4);
                            ubv=min(32*t3+31,t4+_nX-1);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (t4-t6+2*_nY)][ (-t4+t7)][0], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (-t4+t7)][3], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (-t4+t7)][4], grid[ t4 % 2][ (t4-t6+2*_nY)][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][1], grid[ t4 % 2][ (t4-t6+2*_nY)][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][2], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][5], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][6], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][7], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][8], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][0], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][3], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][4], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][1], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][2], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][5], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][6], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][7], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][8], ( t4), ( (t4-t6+2*_nY)), ( (-t4+t7)));;
                            }
                        }
                        for (t6=max(max(32*t2,t4),-32*t1+32*t2+2*t4-31); t6<=min(min(32*t2+31,-32*t1+32*t2+2*t4),t4+_nY-1); t6++) {
                            lbv=max(32*t3,t4+1);
                            ubv=min(32*t3+31,t4+_nX);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (-t4+t6)][ (t4-t7+2*_nX)][0], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (t4-t7+2*_nX)][3], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (t4-t7+2*_nX)][4], grid[ t4 % 2][ (-t4+t6)][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][1], grid[ t4 % 2][ (-t4+t6)][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][2], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][5], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][6], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][7], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][8], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][0], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][3], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][4], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][1], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][2], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][5], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][6], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][7], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][8], ( t4), ( (-t4+t6)), ( (t4-t7+2*_nX)));;
                            }
                        }
                        for (t6=max(max(32*t2,t4+1),-32*t1+32*t2+2*t4-31); t6<=min(min(32*t2+31,t4+_nY),-32*t1+32*t2+2*t4); t6++) {
                            lbv=max(32*t3,t4+1);
                            ubv=min(32*t3+31,t4+_nX);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][0], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (t4-t7+2*_nX)][3], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (t4-t7+2*_nX)][4], grid[ t4 % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][1], grid[ t4 % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][2], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][5], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][6], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][7], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][8], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][0], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][3], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][4], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][1], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][2], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][5], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][6], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][7], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][8], ( t4), ( (t4-t6+2*_nY)), ( (t4-t7+2*_nX)));;
                            }
                        }
                    }
                    if ((t1 >= max(ceild(32*t2+32*t3-_nY+1,32),2*t3)) && (t2 <= floord(32*t3+_nY+30,32)) && (t2 >= t3+1) && (t3 <= floord(_nTimesteps-32,32))) {
                        for (t6=max(32*t2,-32*t1+32*t2+64*t3+31); t6<=min(min(32*t2+31,32*t3+_nY+30),-32*t1+32*t2+64*t3+62); t6++) {
                            lbm_kernel(grid[ (32*t3+31) % 2][ (-32*t3+t6-31)][ 0][0], grid[ (32*t3+31) % 2][ (-32*t3+t6-31) == 0 ? 2 * _nY - 1 : (-32*t3+t6-31) - 1][ 0][3], grid[ (32*t3+31) % 2][ (-32*t3+t6-31) + 1 == 2 * _nY ? 0 : (-32*t3+t6-31) + 1][ 0][4], grid[ (32*t3+31) % 2][ (-32*t3+t6-31)][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (32*t3+31) % 2][ (-32*t3+t6-31)][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][2], grid[ (32*t3+31) % 2][ (-32*t3+t6-31) == 0 ? 2 * _nY - 1 : (-32*t3+t6-31) - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (32*t3+31) % 2][ (-32*t3+t6-31) == 0 ? 2 * _nY - 1 : (-32*t3+t6-31) - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (32*t3+31) % 2][ (-32*t3+t6-31) + 1 == 2 * _nY ? 0 : (-32*t3+t6-31) + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][7], grid[ (32*t3+31) % 2][ (-32*t3+t6-31) + 1 == 2 * _nY ? 0 : (-32*t3+t6-31) + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][8], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][0], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][3], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][4], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][1], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][2], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][5], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][6], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][7], &grid[( (32*t3+31) + 1) % 2][ (-32*t3+t6-31)][ 0][8], ( (32*t3+31)), ( (-32*t3+t6-31)), ( 0));;
                        }
                        for (t6=max(32*t2,-32*t1+32*t2+64*t3+31); t6<=min(min(32*t2+31,32*t3+_nY+31),-32*t1+32*t2+64*t3+62); t6++) {
                            lbm_kernel(grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31)][ 0][0], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY+31) - 1][ 0][3], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY+31) + 1][ 0][4], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31)][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31)][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][2], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY+31) - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY+31) - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY+31) + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][7], grid[ (32*t3+31) % 2][ (32*t3-t6+2*_nY+31) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY+31) + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][8], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][0], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][3], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][4], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][1], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][2], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][5], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][6], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][7], &grid[( (32*t3+31) + 1) % 2][ (32*t3-t6+2*_nY+31)][ 0][8], ( (32*t3+31)), ( (32*t3-t6+2*_nY+31)), ( 0));;
                        }
                    }
                    if ((t1 == 2*t2) && (t1 <= min(floord(_nTimesteps-32,16),2*t3-2)) && (t1 >= ceild(32*t3-_nX-30,16))) {
                        lbv=32*t3;
                        ubv=min(32*t3+31,16*t1+_nX+30);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            if (t1%2 == 0) {
                                lbm_kernel(grid[ (16*t1+31) % 2][ 0][ (-16*t1+t7-31)][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7-31)][3], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7-31)][4], grid[ (16*t1+31) % 2][ 0][ (-16*t1+t7-31) == 0 ? 2 * _nX - 1 : (-16*t1+t7-31) - 1][1], grid[ (16*t1+31) % 2][ 0][ (-16*t1+t7-31) + 1 == 2 * _nX ? 0 : (-16*t1+t7-31) + 1][2], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7-31) == 0 ? 2 * _nX - 1 : (-16*t1+t7-31) - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7-31) + 1 == 2 * _nX ? 0 : (-16*t1+t7-31) + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7-31) == 0 ? 2 * _nX - 1 : (-16*t1+t7-31) - 1][7], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7-31) + 1 == 2 * _nX ? 0 : (-16*t1+t7-31) + 1][8], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][0], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][3], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][4], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][1], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][2], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][5], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][6], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][7], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][8], ( (16*t1+31)), ( 0), ( (-16*t1+t7-31)));;
                            }
                        }
                        lbv=32*t3;
                        ubv=min(32*t3+31,16*t1+_nX+31);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            if (t1%2 == 0) {
                                lbm_kernel(grid[ (16*t1+31) % 2][ 0][ (16*t1-t7+2*_nX+31)][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX+31)][3], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX+31)][4], grid[ (16*t1+31) % 2][ 0][ (16*t1-t7+2*_nX+31) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX+31) - 1][1], grid[ (16*t1+31) % 2][ 0][ (16*t1-t7+2*_nX+31) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX+31) + 1][2], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX+31) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX+31) - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX+31) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX+31) + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX+31) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX+31) - 1][7], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX+31) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX+31) + 1][8], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][0], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][3], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][4], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][1], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][2], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][5], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][6], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][7], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][8], ( (16*t1+31)), ( 0), ( (16*t1-t7+2*_nX+31)));;
                            }
                        }
                    }
                    if ((t1 == 2*t2) && (t1 == 2*t3) && (t1 <= floord(_nTimesteps-32,16))) {
                        if (t1%2 == 0) {
                            lbm_kernel(grid[ (16*t1+31) % 2][ 0][ 0][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ 0][3], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ 0][4], grid[ (16*t1+31) % 2][ 0][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (16*t1+31) % 2][ 0][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][2], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][7], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][8], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][0], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][3], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][4], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][1], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][2], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][5], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][6], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][7], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][8], ( (16*t1+31)), ( 0), ( 0));;
                        }
                    }
                    if ((t1 <= min(min(floord(32*t2-_nY,16),floord(32*t2+_nTimesteps-_nY-32,32)),floord(32*t2+32*t3-_nY-1,32))) && (t1 >= ceild(32*t2+32*t3-_nY-_nX-30,32))) {
                        lbv=max(32*t3,32*t1-32*t2+_nY+31);
                        ubv=min(32*t3+31,32*t1-32*t2+_nY+_nX+30);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][0], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t1+32*t2+t7-_nY-31)][3], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t1+32*t2+t7-_nY-31)][4], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31) == 0 ? 2 * _nX - 1 : (-32*t1+32*t2+t7-_nY-31) - 1][1], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31) + 1 == 2 * _nX ? 0 : (-32*t1+32*t2+t7-_nY-31) + 1][2], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t1+32*t2+t7-_nY-31) == 0 ? 2 * _nX - 1 : (-32*t1+32*t2+t7-_nY-31) - 1][5], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t1+32*t2+t7-_nY-31) + 1 == 2 * _nX ? 0 : (-32*t1+32*t2+t7-_nY-31) + 1][6], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t1+32*t2+t7-_nY-31) == 0 ? 2 * _nX - 1 : (-32*t1+32*t2+t7-_nY-31) - 1][7], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t1+32*t2+t7-_nY-31) + 1 == 2 * _nX ? 0 : (-32*t1+32*t2+t7-_nY-31) + 1][8], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][0], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][3], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][4], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][1], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][2], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][5], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][6], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][7], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (-32*t1+32*t2+t7-_nY-31)][8], ( (32*t1-32*t2+_nY+31)), ( _nY), ( (-32*t1+32*t2+t7-_nY-31)));;
                        }
                        lbv=max(32*t3,32*t1-32*t2+_nY+32);
                        ubv=min(32*t3+31,32*t1-32*t2+_nY+_nX+31);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][0], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t1-32*t2-t7+_nY+2*_nX+31)][3], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t1-32*t2-t7+_nY+2*_nX+31)][4], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31) == 0 ? 2 * _nX - 1 : (32*t1-32*t2-t7+_nY+2*_nX+31) - 1][1], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31) + 1 == 2 * _nX ? 0 : (32*t1-32*t2-t7+_nY+2*_nX+31) + 1][2], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t1-32*t2-t7+_nY+2*_nX+31) == 0 ? 2 * _nX - 1 : (32*t1-32*t2-t7+_nY+2*_nX+31) - 1][5], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t1-32*t2-t7+_nY+2*_nX+31) + 1 == 2 * _nX ? 0 : (32*t1-32*t2-t7+_nY+2*_nX+31) + 1][6], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t1-32*t2-t7+_nY+2*_nX+31) == 0 ? 2 * _nX - 1 : (32*t1-32*t2-t7+_nY+2*_nX+31) - 1][7], grid[ (32*t1-32*t2+_nY+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t1-32*t2-t7+_nY+2*_nX+31) + 1 == 2 * _nX ? 0 : (32*t1-32*t2-t7+_nY+2*_nX+31) + 1][8], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][0], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][3], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][4], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][1], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][2], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][5], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][6], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][7], &grid[( (32*t1-32*t2+_nY+31) + 1) % 2][ _nY][ (32*t1-32*t2-t7+_nY+2*_nX+31)][8], ( (32*t1-32*t2+_nY+31)), ( _nY), ( (32*t1-32*t2-t7+_nY+2*_nX+31)));;
                        }
                    }
                    if ((t1 == 2*t3) && (t1 <= floord(_nTimesteps-32,16)) && (16*t1 == 32*t2-_nY)) {
                        if ((16*t1+31*_nY)%32 == 0) {
                            if (_nY%32 == 0) {
                                lbm_kernel(grid[ (16*t1+31) % 2][ _nY][ 0][0], grid[ (16*t1+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0][3], grid[ (16*t1+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0][4], grid[ (16*t1+31) % 2][ _nY][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (16*t1+31) % 2][ _nY][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][2], grid[ (16*t1+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (16*t1+31) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (16*t1+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][7], grid[ (16*t1+31) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][8], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][0], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][3], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][4], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][1], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][2], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][5], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][6], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][7], &grid[( (16*t1+31) + 1) % 2][ _nY][ 0][8], ( (16*t1+31)), ( _nY), ( 0));;
                            }
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

    printf("\tTime taken : %7.5lfs\n", tdiff);
    printf("\tMLUPS      : %7.5lf\n", (total_lattice_pts / (1.0e6 * tdiff)));
#endif

#ifdef DEBUG
    /* Dump rho, uX, uY for the entire domain to verify results */
    dumpVelocities(t);
#endif

    return 0;
}

/* Performs LBM kernel for one lattice cell */

void lbm_kernel(double in_c, double in_n, double in_s, double in_e, double in_w,
                double in_ne, double in_nw, double in_se, double in_sw,
                double *restrict out_c, double *restrict out_n,
                double *restrict out_s, double *restrict out_e,
                double *restrict out_w, double *restrict out_ne,
                double *restrict out_nw, double *restrict out_se,
                double *restrict out_sw, int t, int y, int x) {
    /* %%INIT%% */
    double lbm_sum_e, lbm_sum_w, lbm_sum_n, lbm_sum_s, lbm_rho, lbm_inv_rho;
    double lbm_uX, lbm_uY, lbm_uX2, lbm_uY2, lbm_u2, lbm_one_minus_u2;
    double lbm_omega_w1_rho, lbm_omega_w2_rho, lbm_omega_w3_rho;
    double lbm_30_uY, lbm_45_uY2, lbm_30_uX, lbm_45_uX2;

    double lbm_out_c, lbm_out_n, lbm_out_s, lbm_out_e, lbm_out_w, lbm_out_ne,
           lbm_out_nw, lbm_out_se, lbm_out_sw;
    double lbm_uX_plus_uY, lbm_uX_minus_uY, lbm_30_uX_plus_uY, lbm_30_uX_minus_uY,
           lbm_45_uX_plus_uY, lbm_45_uX_minus_uY;
    double position;
    /* %%INIT%% */

    /* #<{(| %%BOUNDARY%% |)}># */
    /* if ((y == 3 || y == nY + 2) && (x > 2 && x < nX + 1) ) { */
    /*   #<{(| Bounce back boundary condition at walls and cylinder boundary |)}># */
    /*   *out_c = in_c; */
    /*  */
    /*   *out_s = in_n; */
    /*   *out_n = in_s; */
    /*   *out_w = in_e; */
    /*   *out_e = in_w; */
    /*  */
    /*   *out_sw = in_ne; */
    /*   *out_se = in_nw; */
    /*   *out_nw = in_se; */
    /*   *out_ne = in_sw; */
    /*  */
    /*   return; */
    /* } */
    /* %%BOUNDARY%% */


    /* %%BOUNDARY%% */
    /* Inlet pressue boundary conditions */
    /* if ((x == 2) && (y == 3)) { */
    /*   #<{(| Bottom left point |)}># */
    /*   in_e = in_w; */
    /*   in_n = in_s; */
    /*   in_ne = in_sw; */
    /*   in_nw = (1.0 / 2.0) * */
    /*           (rho_in - (in_c + in_e + in_w + in_n + in_s + in_ne + in_sw)); */
    /*   in_se = (1.0 / 2.0) * */
    /*           (rho_in - (in_c + in_e + in_w + in_n + in_s + in_ne + in_sw)); */
    /* } */
    /*  */
    /* if ((x == 2) && (y == nY + 2)) { */
    /*   #<{(| Top left point |)}># */
    /*   in_e = in_w; */
    /*   in_s = in_n; */
    /*   in_se = in_nw; */
    /*  */
    /*   in_sw = (1.0 / 2.0) * */
    /*           (rho_in - (in_c + in_e + in_w + in_n + in_s + in_se + in_nw)); */
    /*   in_ne = (1.0 / 2.0) * */
    /*           (rho_in - (in_c + in_e + in_w + in_n + in_s + in_se + in_nw)); */
    /* } */

    /* #<{(| Outlet pressure boundary conditions |)}># */
    /* if ((x == nX + 1) && (y == 3)) { */
    /*   #<{(| Bottom right point |)}># */
    /*   in_w = in_e; */
    /*   in_n = in_s; */
    /*   in_nw = in_se; */
    /*  */
    /*   in_ne = (1.0 / 2.0) * */
    /*           (rho_out - (in_c + in_e + in_w + in_n + in_s + in_nw + in_se)); */
    /*   in_sw = (1.0 / 2.0) * */
    /*           (rho_out - (in_c + in_e + in_w + in_n + in_s + in_nw + in_se)); */
    /* } */
    /*  */
    /* if ((x == nX + 1) && (y == 3)) { */
    /*   #<{(| Top right point |)}># */
    /*   in_w = in_e; */
    /*   in_s = in_n; */
    /*   in_sw = in_ne; */
    /*  */
    /*   in_nw = (1.0 / 2.0) * */
    /*           (rho_out - (in_c + in_e + in_w + in_n + in_s + in_ne + in_sw)); */
    /*   in_se = (1.0 / 2.0) * */
    /*           (rho_out - (in_c + in_e + in_w + in_n + in_s + in_ne + in_sw)); */
    /* } */
    /* #<{(| %%BOUNDARY%% |)}># */

    lbm_sum_e = in_ne + in_e + in_se;
    lbm_sum_w = in_nw + in_w + in_sw;
    lbm_sum_n = in_nw + in_n + in_ne;
    lbm_sum_s = in_sw + in_s + in_se;

    /* Compute rho */
    lbm_rho = lbm_sum_e + (in_n + in_c + in_s) + lbm_sum_w;
    lbm_inv_rho = 1.0 / lbm_rho;

    /* Compute velocity along x-axis */
    lbm_uX = lbm_inv_rho * (lbm_sum_e - lbm_sum_w);

    /* Compute velocity along y-axis */
    lbm_uY = lbm_inv_rho * (lbm_sum_n - lbm_sum_s);

    /* Apply Zou-He pressure boundary conditions */
    /* Inlet */
    /* %%BOUNDARY%% */
    if ((x == 0)) {
        lbm_uX =
            1.0 - ((in_c + in_n + in_s + 2.0 * (in_nw + in_w + in_sw)) / rho_in);
        lbm_uY = 0.0;
        lbm_rho = rho_in;

        in_e = in_w + (2.0 / 3.0) * (rho_in * lbm_uX);

        in_ne = in_sw - (1.0 / 2.0) * (in_n - in_s) + (1.0 / 6.0) * rho_in * lbm_uX;

        in_se = in_nw + (1.0 / 2.0) * (in_n - in_s) + (1.0 / 6.0) * rho_in * lbm_uX;
    }

    /* Outlet */
    if ((x == nX - 1)) {
        lbm_uX =
            1.0 - ((in_c + in_n + in_s + 2.0 * (in_ne + in_e + in_se)) / rho_out);
        lbm_uY = 0.0;
        lbm_rho = rho_out;

        in_w = in_e - (2.0 / 3.0) * rho_out * lbm_uX;

        in_nw =
            in_se - (1.0 / 6.0) * rho_out * lbm_uX - (1.0 / 2.0) * (in_n - in_s);

        in_sw =
            in_ne - (1.0 / 6.0) * rho_out * lbm_uX + (1.0 / 2.0) * (in_n - in_s);
    }

    /* %%BOUNDARY%% */
    lbm_uX2 = lbm_uX * lbm_uX;
    lbm_uY2 = lbm_uY * lbm_uY;

    lbm_u2 = 1.5 * (lbm_uX2 + lbm_uY2);

    lbm_one_minus_u2 = 1.0 - lbm_u2;

    lbm_omega_w1_rho = omega * w1 * lbm_rho;
    lbm_omega_w2_rho = omega * w2 * lbm_rho;
    lbm_omega_w3_rho = omega * w3 * lbm_rho;

    lbm_30_uY = 3.0 * lbm_uY;
    lbm_45_uY2 = 4.5 * lbm_uY2;
    lbm_30_uX = 3.0 * lbm_uX;
    lbm_45_uX2 = 4.5 * lbm_uX2;

    lbm_uX_plus_uY = lbm_uX + lbm_uY;
    lbm_uX_minus_uY = lbm_uX - lbm_uY;

    lbm_30_uX_plus_uY = 3.0 * lbm_uX_plus_uY;
    lbm_30_uX_minus_uY = 3.0 * lbm_uX_minus_uY;
    lbm_45_uX_plus_uY = 4.5 * lbm_uX_plus_uY * lbm_uX_plus_uY;
    lbm_45_uX_minus_uY = 4.5 * lbm_uX_minus_uY * lbm_uX_minus_uY;

    /* Compute and keep new PDFs */
    lbm_out_c = one_minus_omega * in_c + lbm_omega_w1_rho * (lbm_one_minus_u2);

    lbm_out_n = one_minus_omega * in_n +
                lbm_omega_w2_rho * (lbm_one_minus_u2 + lbm_30_uY + lbm_45_uY2);
    lbm_out_s = one_minus_omega * in_s +
                lbm_omega_w2_rho * (lbm_one_minus_u2 - lbm_30_uY + lbm_45_uY2);
    lbm_out_e = one_minus_omega * in_e +
                lbm_omega_w2_rho * (lbm_one_minus_u2 + lbm_30_uX + lbm_45_uX2);
    lbm_out_w = one_minus_omega * in_w +
                lbm_omega_w2_rho * (lbm_one_minus_u2 - lbm_30_uX + lbm_45_uX2);

    lbm_out_ne = one_minus_omega * in_ne +
                 lbm_omega_w3_rho *
                 (lbm_one_minus_u2 + lbm_30_uX_plus_uY + lbm_45_uX_plus_uY);
    lbm_out_nw = one_minus_omega * in_nw +
                 lbm_omega_w3_rho *
                 (lbm_one_minus_u2 - lbm_30_uX_minus_uY + lbm_45_uX_minus_uY);
    lbm_out_se = one_minus_omega * in_se +
                 lbm_omega_w3_rho *
                 (lbm_one_minus_u2 + lbm_30_uX_minus_uY + lbm_45_uX_minus_uY);
    lbm_out_sw = one_minus_omega * in_sw +
                 lbm_omega_w3_rho *
                 (lbm_one_minus_u2 - lbm_30_uX_plus_uY + lbm_45_uX_plus_uY);

    *out_c = lbm_out_c;

    *out_n = lbm_out_n;
    *out_s = lbm_out_s;
    *out_e = lbm_out_e;
    *out_w = lbm_out_w;

    *out_ne = lbm_out_ne;
    *out_nw = lbm_out_nw;
    *out_se = lbm_out_se;
    *out_sw = lbm_out_sw;
    /* u2 = 1.5 * (uX * uX + uY * uY); */
    /*  */
    /* #<{(| Compute and keep new PDFs |)}># */
    /* *out_c = (1.0 - omega) * in_c + (omega * (w1 * rho * (1.0 - u2))); */
    /*  */
    /* *out_n = (1.0 - omega) * in_n + */
    /*          (omega * (w2 * rho * (1.0 + 3.0 * uY + 4.5 * uY * uY - u2))); */
    /* *out_s = (1.0 - omega) * in_s + */
    /*          (omega * (w2 * rho * (1.0 - 3.0 * uY + 4.5 * uY * uY - u2))); */
    /* *out_e = (1.0 - omega) * in_e + */
    /*          (omega * (w2 * rho * (1.0 + 3.0 * uX + 4.5 * uX * uX - u2))); */
    /* *out_w = (1.0 - omega) * in_w + */
    /*          (omega * (w2 * rho * (1.0 - 3.0 * uX + 4.5 * uX * uX - u2))); */
    /*  */
    /* *out_ne = (1.0 - omega) * in_ne + (omega * (w3 * rho *  */
    /*               (1.0 + 3.0 * (uX + uY) + 4.5 * (uX + uY) * (uX + uY) - u2)));
     *                  */
    /* *out_nw = (1.0 - omega) * in_nw + (omega * (w3 * rho *  */
    /*               (1.0 + 3.0 * (-uX + uY) + 4.5 * (-uX + uY) * (-uX + uY) -
     *                  * u2))); */
    /* *out_se = (1.0 - omega) * in_se + (omega * (w3 * rho *  */
    /*               (1.0 + 3.0 * (uX - uY) + 4.5 * (uX - uY) * (uX - uY) - u2)));
     *                  */
    /* *out_sw = (1.0 - omega) * in_sw + (omega * (w3 * rho *  */
    /*               (1.0 + 3.0 * (-uX - uY) + 4.5 * (-uX - uY) * (-uX - uY) -
     *                  * u2))); */
}
void dumpVelocities(int t) {
    int y, x;
    double rho, uX, uY;

    for (y = 1; y < nY-1 ; y++) {
        for (x = 1; x < nX-1; x++) {
            /* Compute rho */
            rho = grid[t % 2][y - 1][x + 0][N] + grid[t % 2][y + 1][x + 0][S] +
                  grid[t % 2][y + 0][x - 1][E] + grid[t % 2][y + 0][x + 1][W] +
                  grid[t % 2][y - 1][x - 1][NE] + grid[t % 2][y - 1][x + 1][NW] +
                  grid[t % 2][y + 1][x - 1][SE] + grid[t % 2][y + 1][x + 1][SW] +
                  grid[t % 2][y + 0][x + 0][C];

            /* Compute velocity along x-axis */
            uX = (grid[t % 2][y + 1][x - 1][SE] + grid[t % 2][y + 0][x - 1][E] +
                  grid[t % 2][y - 1][x - 1][NE]) -
                 (grid[t % 2][y + 1][x + 1][SW] + grid[t % 2][y + 0][x + 1][W] +
                  grid[t % 2][y - 1][x + 1][NW]);
            uX = uX / rho;

            /* Compute velocity along y-axis */
            uY = (grid[t % 2][y - 1][x + 1][NW] + grid[t % 2][y - 1][x + 0][N] +
                  grid[t % 2][y - 1][x - 1][NE]) -
                 (grid[t % 2][y + 1][x + 1][SW] + grid[t % 2][y + 1][x + 0][S] +
                  grid[t % 2][y + 1][x - 1][SE]);
            uY = uY / rho;

            fprintf(stderr, "%0.6lf,%0.6lf,%0.6lf\n", rho, uX, uY);
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

/* icc -O3 -fp-model precise -restrict -DDEBUG -DTIME poiseuille_d2q9.c -o
 * poiseuille_d2q9.elf */
