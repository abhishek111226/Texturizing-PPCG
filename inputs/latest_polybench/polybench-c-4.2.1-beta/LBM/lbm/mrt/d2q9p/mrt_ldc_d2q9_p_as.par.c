#include <omp.h>
#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/****************************************************************************/
/* Lid Driven Cavity(LDC) - 2D simulation using MRT-LBM with a D2Q9 lattice */
/*                                                                          */
/* Scheme: 2-Grid, Pull, Compute, Keep                                      */
/*                                                                          */
/* Lattice naming convention                                                */
/*   c6[NW]  c2[N]  c5[NE]                                                  */
/*   c3[W]   c0[C]  c1[E]                                                   */
/*   c7[SW]  c4[S]  c8[SE]                                                  */
/*                                                                          */
/* Author: Irshad Pananilath <pmirshad+code@gmail.com>                      */
/*                                                                          */
/****************************************************************************/

#include <stdio.h>
#include <sys/time.h>

#ifndef nX
#define nX 1024  // Size of domain along x-axis
#endif

#ifndef nY
#define nY 1024  // Size of domain along y-axis
#endif

#ifndef nK
#define nK 9  // D2Q9
#endif

#ifndef nTimesteps
#define nTimesteps 20000  // Number of timesteps for the simulation
#endif

#define C 0
#define N 2
#define S 4
#define E 1
#define W 3
#define NE 5
#define NW 6
#define SE 8
#define SW 7

/* Tunable parameters */
const double re = 1000;
const double uTop = 0.1;

/* Calculated parameters: Set in main */
double omega = 0.0;
double nu = 0.0;

/* Moment relaxation rates */
double s_0 = 0.0;
double s_1 = 0.0;
double s_2 = 0.0;
double s_3 = 0.0;
double s_4 = 0.0;
double s_5 = 0.0;
double s_6 = 0.0;
double s_7 = 0.0;
double s_8 = 0.0;

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
    omega = 2.0 / ((6.0 * uTop * (nX - 0.5) / re) + 1.0);
    nu = (1.0 / omega - 0.5) / 3.0;

    s_0 = 1.0;
    s_1 = 1.4;
    s_2 = 1.4;
    s_3 = 1.0;
    s_4 = 1.2;
    s_5 = 1.0;
    s_6 = 1.2;
    /* s_7 = omega; */
    /* s_8 = omega; */
    s_7 = 2.0 / (1.0 + 6.0 * nu);
    s_8 = 2.0 / (1.0 + 6.0 * nu);

    printf(
        "2D Lid Driven Cavity simulation using MRT-LBM with D2Q9 lattice:\n"
        "\tscheme     : 2-Grid, Fused, Pull\n"
        "\tgrid size  : %d x %d = %.2lf * 10^3 Cells\n"
        "\tnTimeSteps : %d\n"
        "\tRe         : %.2lf\n"
        "\tuTop       : %.6lf\n"
        "\tomega      : %.6lf\n",
        nX, nY, nX * nY / 1.0e3, nTimesteps, re, uTop, omega);

    /* Initialize the PDFs to equilibrium values */
    /* Calculate initial equilibrium moments */
    for (y = 0; y < nY; y++) {
        for (x = 0; x < nX; x++) {
            grid[0][y][x][0] = w1;
            grid[1][y][x][0] = w1;

            for (k = 1; k < 5; k++) {
                grid[0][y][x][k] = w2;
                grid[1][y][x][k] = w2;
            }

            for (k = 5; k < nK; k++) {
                grid[0][y][x][k] = w3;
                grid[1][y][x][k] = w3;
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
        for (t1=0; t1<=min(floord(32*floord(_nTimesteps+31,32)+61*_nTimesteps+31*_nY-93,992),floord(174752*floord(_nTimesteps+31,32)+5461*_nTimesteps+5456*_nY-16378,174752)); t1++) {
            lbp=max(ceild(t1,2),ceild(32*t1-_nTimesteps+1,32));
            ubp=min(min(floord(_nTimesteps+_nY-1,32),floord(32*t1+_nY+31,64)),t1);
            #pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7)
            for (t2=lbp; t2<=ubp; t2++) {
                for (t3=t1-t2; t3<=min(floord(_nTimesteps+_nX-1,32),floord(32*t1-32*t2+_nX+31,32)); t3++) {
                    if ((t1 <= floord(64*t2-_nY,32)) && (t2 <= floord(32*t3+_nY+30,32)) && (t2 >= ceild(32*t3+_nY-_nX+1,32))) {
                        lbv=max(32*t3,32*t2-_nY);
                        ubv=min(32*t3+31,32*t2-_nY+_nX-1);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t2-_nY) % 2][ _nY][ (-32*t2+t7+_nY)][0], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t2+t7+_nY)][2], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t2+t7+_nY)][4], grid[ (32*t2-_nY) % 2][ _nY][ (-32*t2+t7+_nY) == 0 ? 2 * _nX - 1 : (-32*t2+t7+_nY) - 1][1], grid[ (32*t2-_nY) % 2][ _nY][ (-32*t2+t7+_nY) + 1 == 2 * _nX ? 0 : (-32*t2+t7+_nY) + 1][3], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t2+t7+_nY) == 0 ? 2 * _nX - 1 : (-32*t2+t7+_nY) - 1][5], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (-32*t2+t7+_nY) + 1 == 2 * _nX ? 0 : (-32*t2+t7+_nY) + 1][6], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t2+t7+_nY) == 0 ? 2 * _nX - 1 : (-32*t2+t7+_nY) - 1][8], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (-32*t2+t7+_nY) + 1 == 2 * _nX ? 0 : (-32*t2+t7+_nY) + 1][7], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][0], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][2], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][4], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][1], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][3], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][5], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][6], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][8], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (-32*t2+t7+_nY)][7], ( (32*t2-_nY)), ( _nY), ( (-32*t2+t7+_nY)));;
                        }
                        lbv=max(32*t3,32*t2-_nY+1);
                        ubv=min(32*t3+31,32*t2-_nY+_nX);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t2-_nY) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][0], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t2-t7-_nY+2*_nX)][2], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t2-t7-_nY+2*_nX)][4], grid[ (32*t2-_nY) % 2][ _nY][ (32*t2-t7-_nY+2*_nX) == 0 ? 2 * _nX - 1 : (32*t2-t7-_nY+2*_nX) - 1][1], grid[ (32*t2-_nY) % 2][ _nY][ (32*t2-t7-_nY+2*_nX) + 1 == 2 * _nX ? 0 : (32*t2-t7-_nY+2*_nX) + 1][3], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t2-t7-_nY+2*_nX) == 0 ? 2 * _nX - 1 : (32*t2-t7-_nY+2*_nX) - 1][5], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ (32*t2-t7-_nY+2*_nX) + 1 == 2 * _nX ? 0 : (32*t2-t7-_nY+2*_nX) + 1][6], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t2-t7-_nY+2*_nX) == 0 ? 2 * _nX - 1 : (32*t2-t7-_nY+2*_nX) - 1][8], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ (32*t2-t7-_nY+2*_nX) + 1 == 2 * _nX ? 0 : (32*t2-t7-_nY+2*_nX) + 1][7], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][0], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][2], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][4], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][1], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][3], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][5], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][6], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][8], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ (32*t2-t7-_nY+2*_nX)][7], ( (32*t2-_nY)), ( _nY), ( (32*t2-t7-_nY+2*_nX)));;
                        }
                    }
                    if ((32*t1 == 64*t2-_nY-31) && (32*t1 == 64*t3+_nY+31)) {
                        if ((_nY+1)%2 == 0) {
                            if ((32*t1+63*_nY+33)%64 == 0) {
                                lbm_kernel(grid[ ((32*t1-_nY+31)/2) % 2][ _nY][ 0][0], grid[ ((32*t1-_nY+31)/2) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0][2], grid[ ((32*t1-_nY+31)/2) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0][4], grid[ ((32*t1-_nY+31)/2) % 2][ _nY][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ ((32*t1-_nY+31)/2) % 2][ _nY][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][3], grid[ ((32*t1-_nY+31)/2) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ ((32*t1-_nY+31)/2) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ ((32*t1-_nY+31)/2) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][8], grid[ ((32*t1-_nY+31)/2) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][7], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][0], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][2], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][4], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][1], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][3], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][5], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][6], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][8], &grid[( ((32*t1-_nY+31)/2) + 1) % 2][ _nY][ 0][7], ( ((32*t1-_nY+31)/2)), ( _nY), ( 0));;
                            }
                        }
                    }
                    if ((t1 <= floord(32*t2+32*t3-_nX,32)) && (t2 <= floord(32*t3+_nY-_nX-1,32)) && (t2 >= ceild(32*t3-_nX-30,32))) {
                        for (t6=max(32*t2,32*t3-_nX); t6<=min(32*t2+31,32*t3+_nY-_nX-1); t6++) {
                            lbm_kernel(grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX)][ _nX][0], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) == 0 ? 2 * _nY - 1 : (-32*t3+t6+_nX) - 1][ _nX][2], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) + 1 == 2 * _nY ? 0 : (-32*t3+t6+_nX) + 1][ _nX][4], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX)][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX)][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][3], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) == 0 ? 2 * _nY - 1 : (-32*t3+t6+_nX) - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) == 0 ? 2 * _nY - 1 : (-32*t3+t6+_nX) - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) + 1 == 2 * _nY ? 0 : (-32*t3+t6+_nX) + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][8], grid[ (32*t3-_nX) % 2][ (-32*t3+t6+_nX) + 1 == 2 * _nY ? 0 : (-32*t3+t6+_nX) + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][7], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][0], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][2], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][4], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][1], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][3], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][5], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][6], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][8], &grid[( (32*t3-_nX) + 1) % 2][ (-32*t3+t6+_nX)][ _nX][7], ( (32*t3-_nX)), ( (-32*t3+t6+_nX)), ( _nX));;
                        }
                        for (t6=max(32*t2,32*t3-_nX+1); t6<=min(32*t2+31,32*t3+_nY-_nX); t6++) {
                            lbm_kernel(grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][0], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY-_nX) - 1][ _nX][2], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY-_nX) + 1][ _nX][4], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][3], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY-_nX) - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) == 0 ? 2 * _nY - 1 : (32*t3-t6+2*_nY-_nX) - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY-_nX) + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][8], grid[ (32*t3-_nX) % 2][ (32*t3-t6+2*_nY-_nX) + 1 == 2 * _nY ? 0 : (32*t3-t6+2*_nY-_nX) + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][7], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][0], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][2], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][4], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][1], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][3], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][5], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][6], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][8], &grid[( (32*t3-_nX) + 1) % 2][ (32*t3-t6+2*_nY-_nX)][ _nX][7], ( (32*t3-_nX)), ( (32*t3-t6+2*_nY-_nX)), ( _nX));;
                        }
                    }
                    if ((t1 == 2*t2) && (16*t1 == 32*t3-_nX-31)) {
                        if (t1%2 == 0) {
                            if ((16*t1+_nX+31)%32 == 0) {
                                lbm_kernel(grid[ (16*t1+31) % 2][ 0][ _nX][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX][2], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX][4], grid[ (16*t1+31) % 2][ 0][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (16*t1+31) % 2][ 0][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][3], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][8], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][7], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][0], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][2], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][4], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][1], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][3], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][5], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][6], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][8], &grid[( (16*t1+31) + 1) % 2][ 0][ _nX][7], ( (16*t1+31)), ( 0), ( _nX));;
                            }
                        }
                    }
                    if ((t1 <= floord(64*t2-_nY,32)) && (32*t2 == 32*t3+_nY-_nX)) {
                        if ((31*_nY+_nX)%32 == 0) {
                            lbm_kernel(grid[ (32*t2-_nY) % 2][ _nY][ _nX][0], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX][2], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX][4], grid[ (32*t2-_nY) % 2][ _nY][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][1], grid[ (32*t2-_nY) % 2][ _nY][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][3], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][5], grid[ (32*t2-_nY) % 2][ _nY == 0 ? 2 * _nY - 1 : _nY - 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][6], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX == 0 ? 2 * _nX - 1 : _nX - 1][8], grid[ (32*t2-_nY) % 2][ _nY + 1 == 2 * _nY ? 0 : _nY + 1][ _nX + 1 == 2 * _nX ? 0 : _nX + 1][7], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][0], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][2], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][4], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][1], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][3], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][5], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][6], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][8], &grid[( (32*t2-_nY) + 1) % 2][ _nY][ _nX][7], ( (32*t2-_nY)), ( _nY), ( _nX));;
                        }
                    }
                    for (t4=max(max(32*t1-32*t2,32*t2-_nY+1),32*t3-_nX+1); t4<=min(min(min(_nTimesteps-1,32*t2+30),32*t3+30),32*t1-32*t2+31); t4++) {
                        for (t6=max(32*t2,t4); t6<=min(32*t2+31,t4+_nY-1); t6++) {
                            lbv=max(32*t3,t4);
                            ubv=min(32*t3+31,t4+_nX-1);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (-t4+t6)][ (-t4+t7)][0], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (-t4+t7)][2], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (-t4+t7)][4], grid[ t4 % 2][ (-t4+t6)][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][1], grid[ t4 % 2][ (-t4+t6)][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][3], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][5], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][6], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][8], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][7], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][0], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][2], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][4], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][1], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][3], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][5], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][6], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][8], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][7], ( t4), ( (-t4+t6)), ( (-t4+t7)));;
                            }
                        }
                        for (t6=max(32*t2,t4+1); t6<=min(32*t2+31,t4+_nY); t6++) {
                            lbv=max(32*t3,t4);
                            ubv=min(32*t3+31,t4+_nX-1);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (t4-t6+2*_nY)][ (-t4+t7)][0], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (-t4+t7)][2], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (-t4+t7)][4], grid[ t4 % 2][ (t4-t6+2*_nY)][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][1], grid[ t4 % 2][ (t4-t6+2*_nY)][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][3], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][5], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][6], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (-t4+t7) == 0 ? 2 * _nX - 1 : (-t4+t7) - 1][8], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (-t4+t7) + 1 == 2 * _nX ? 0 : (-t4+t7) + 1][7], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][0], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][2], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][4], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][1], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][3], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][5], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][6], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][8], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (-t4+t7)][7], ( t4), ( (t4-t6+2*_nY)), ( (-t4+t7)));;
                            }
                        }
                        for (t6=max(32*t2,t4); t6<=min(32*t2+31,t4+_nY-1); t6++) {
                            lbv=max(32*t3,t4+1);
                            ubv=min(32*t3+31,t4+_nX);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (-t4+t6)][ (t4-t7+2*_nX)][0], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (t4-t7+2*_nX)][2], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (t4-t7+2*_nX)][4], grid[ t4 % 2][ (-t4+t6)][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][1], grid[ t4 % 2][ (-t4+t6)][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][3], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][5], grid[ t4 % 2][ (-t4+t6) == 0 ? 2 * _nY - 1 : (-t4+t6) - 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][6], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][8], grid[ t4 % 2][ (-t4+t6) + 1 == 2 * _nY ? 0 : (-t4+t6) + 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][7], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][0], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][2], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][4], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][1], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][3], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][5], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][6], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][8], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+2*_nX)][7], ( t4), ( (-t4+t6)), ( (t4-t7+2*_nX)));;
                            }
                        }
                        for (t6=max(32*t2,t4+1); t6<=min(32*t2+31,t4+_nY); t6++) {
                            lbv=max(32*t3,t4+1);
                            ubv=min(32*t3+31,t4+_nX);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][0], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (t4-t7+2*_nX)][2], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (t4-t7+2*_nX)][4], grid[ t4 % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][1], grid[ t4 % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][3], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][5], grid[ t4 % 2][ (t4-t6+2*_nY) == 0 ? 2 * _nY - 1 : (t4-t6+2*_nY) - 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][6], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (t4-t7+2*_nX) == 0 ? 2 * _nX - 1 : (t4-t7+2*_nX) - 1][8], grid[ t4 % 2][ (t4-t6+2*_nY) + 1 == 2 * _nY ? 0 : (t4-t6+2*_nY) + 1][ (t4-t7+2*_nX) + 1 == 2 * _nX ? 0 : (t4-t7+2*_nX) + 1][7], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][0], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][2], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][4], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][1], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][3], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][5], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][6], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][8], &grid[( t4 + 1) % 2][ (t4-t6+2*_nY)][ (t4-t7+2*_nX)][7], ( t4), ( (t4-t6+2*_nY)), ( (t4-t7+2*_nX)));;
                            }
                        }
                    }
                    if ((t1 == t2+t3) && (t1 <= min(floord(32*t2+_nTimesteps-32,32),2*t2-1)) && (t1 >= ceild(64*t2-_nY-30,32))) {
                        for (t6=32*t2; t6<=min(32*t2+31,32*t1-32*t2+_nY+30); t6++) {
                            lbm_kernel(grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31)][ 0][0], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31) == 0 ? 2 * _nY - 1 : (-32*t1+32*t2+t6-31) - 1][ 0][2], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31) + 1 == 2 * _nY ? 0 : (-32*t1+32*t2+t6-31) + 1][ 0][4], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31)][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31)][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][3], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31) == 0 ? 2 * _nY - 1 : (-32*t1+32*t2+t6-31) - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31) == 0 ? 2 * _nY - 1 : (-32*t1+32*t2+t6-31) - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31) + 1 == 2 * _nY ? 0 : (-32*t1+32*t2+t6-31) + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][8], grid[ (32*t1-32*t2+31) % 2][ (-32*t1+32*t2+t6-31) + 1 == 2 * _nY ? 0 : (-32*t1+32*t2+t6-31) + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][7], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][0], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][2], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][4], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][1], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][3], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][5], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][6], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][8], &grid[( (32*t1-32*t2+31) + 1) % 2][ (-32*t1+32*t2+t6-31)][ 0][7], ( (32*t1-32*t2+31)), ( (-32*t1+32*t2+t6-31)), ( 0));;
                        }
                        for (t6=32*t2; t6<=min(32*t2+31,32*t1-32*t2+_nY+31); t6++) {
                            lbm_kernel(grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][0], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31) == 0 ? 2 * _nY - 1 : (32*t1-32*t2-t6+2*_nY+31) - 1][ 0][2], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31) + 1 == 2 * _nY ? 0 : (32*t1-32*t2-t6+2*_nY+31) + 1][ 0][4], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][3], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31) == 0 ? 2 * _nY - 1 : (32*t1-32*t2-t6+2*_nY+31) - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31) == 0 ? 2 * _nY - 1 : (32*t1-32*t2-t6+2*_nY+31) - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31) + 1 == 2 * _nY ? 0 : (32*t1-32*t2-t6+2*_nY+31) + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][8], grid[ (32*t1-32*t2+31) % 2][ (32*t1-32*t2-t6+2*_nY+31) + 1 == 2 * _nY ? 0 : (32*t1-32*t2-t6+2*_nY+31) + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][7], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][0], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][2], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][4], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][1], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][3], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][5], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][6], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][8], &grid[( (32*t1-32*t2+31) + 1) % 2][ (32*t1-32*t2-t6+2*_nY+31)][ 0][7], ( (32*t1-32*t2+31)), ( (32*t1-32*t2-t6+2*_nY+31)), ( 0));;
                        }
                    }
                    if ((t1 == 2*t2) && (t1 <= min(floord(_nTimesteps-32,16),2*t3-2)) && (t1 >= ceild(32*t3-_nX-30,16))) {
                        lbv=32*t3;
                        ubv=min(32*t3+31,16*t1+_nX+30);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            if (t1%2 == 0) {
                                lbm_kernel(grid[ (16*t1+31) % 2][ 0][ (-16*t1+t7-31)][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7-31)][2], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7-31)][4], grid[ (16*t1+31) % 2][ 0][ (-16*t1+t7-31) == 0 ? 2 * _nX - 1 : (-16*t1+t7-31) - 1][1], grid[ (16*t1+31) % 2][ 0][ (-16*t1+t7-31) + 1 == 2 * _nX ? 0 : (-16*t1+t7-31) + 1][3], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7-31) == 0 ? 2 * _nX - 1 : (-16*t1+t7-31) - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (-16*t1+t7-31) + 1 == 2 * _nX ? 0 : (-16*t1+t7-31) + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7-31) == 0 ? 2 * _nX - 1 : (-16*t1+t7-31) - 1][8], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (-16*t1+t7-31) + 1 == 2 * _nX ? 0 : (-16*t1+t7-31) + 1][7], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][0], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][2], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][4], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][1], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][3], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][5], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][6], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][8], &grid[( (16*t1+31) + 1) % 2][ 0][ (-16*t1+t7-31)][7], ( (16*t1+31)), ( 0), ( (-16*t1+t7-31)));;
                            }
                        }
                        lbv=32*t3;
                        ubv=min(32*t3+31,16*t1+_nX+31);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            if (t1%2 == 0) {
                                lbm_kernel(grid[ (16*t1+31) % 2][ 0][ (16*t1-t7+2*_nX+31)][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX+31)][2], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX+31)][4], grid[ (16*t1+31) % 2][ 0][ (16*t1-t7+2*_nX+31) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX+31) - 1][1], grid[ (16*t1+31) % 2][ 0][ (16*t1-t7+2*_nX+31) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX+31) + 1][3], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX+31) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX+31) - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ (16*t1-t7+2*_nX+31) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX+31) + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX+31) == 0 ? 2 * _nX - 1 : (16*t1-t7+2*_nX+31) - 1][8], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ (16*t1-t7+2*_nX+31) + 1 == 2 * _nX ? 0 : (16*t1-t7+2*_nX+31) + 1][7], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][0], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][2], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][4], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][1], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][3], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][5], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][6], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][8], &grid[( (16*t1+31) + 1) % 2][ 0][ (16*t1-t7+2*_nX+31)][7], ( (16*t1+31)), ( 0), ( (16*t1-t7+2*_nX+31)));;
                            }
                        }
                    }
                    if ((t1 == 2*t2) && (t1 == 2*t3) && (t1 <= floord(_nTimesteps-32,16))) {
                        if (t1%2 == 0) {
                            lbm_kernel(grid[ (16*t1+31) % 2][ 0][ 0][0], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ 0][2], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ 0][4], grid[ (16*t1+31) % 2][ 0][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][1], grid[ (16*t1+31) % 2][ 0][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][3], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][5], grid[ (16*t1+31) % 2][ 0 == 0 ? 2 * _nY - 1 : 0 - 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][6], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ 0 == 0 ? 2 * _nX - 1 : 0 - 1][8], grid[ (16*t1+31) % 2][ 0 + 1 == 2 * _nY ? 0 : 0 + 1][ 0 + 1 == 2 * _nX ? 0 : 0 + 1][7], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][0], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][2], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][4], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][1], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][3], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][5], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][6], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][8], &grid[( (16*t1+31) + 1) % 2][ 0][ 0][7], ( (16*t1+31)), ( 0), ( 0));;
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

/* Performs LBM kernel for one lattice cell */
void lbm_kernel(double in_c, double in_n, double in_s, double in_e, double in_w,
                double in_ne, double in_nw, double in_se, double in_sw,
                double *restrict out_c, double *restrict out_n,
                double *restrict out_s, double *restrict out_e,
                double *restrict out_w, double *restrict out_ne,
                double *restrict out_nw, double *restrict out_se,
                double *restrict out_sw, int t, int y, int x) {
    /* %%INIT%% */
    double rho, uX, uY, u2;
    double m_eq0, m_eq1, m_eq2, m_eq3, m_eq4, m_eq5, m_eq6, m_eq7, m_eq8;
    double m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8;
    double x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,x16,x17;
    /* %%INIT%% */


    /* #<{(| %%BOUNDARY%% |)}># */
    /* if (((y == 2 || y == nY + 3) && (x > 1 && x < nX + 2)) || */
    /*         ((x == 1 || x == nX + 2) && (y > 2 && y < nY + 3))) { */
    /*     #<{(| Bounce back boundary condition at walls |)}># */
    /*     *out_c = in_c; */
    /*  */
    /*     *out_s = in_n; */
    /*     *out_n = in_s; */
    /*     *out_w = in_e; */
    /*     *out_e = in_w; */
    /*  */
    /*     *out_sw = in_ne; */
    /*     *out_se = in_nw; */
    /*     *out_nw = in_se; */
    /*     *out_ne = in_sw; */
    /*  */
    /*     return; */
    /* } */
    /* %%BOUNDARY%% */

    x0 = in_nw + in_s + in_sw;
    x1 = in_c + in_e + in_n + in_ne + in_se + in_w + x0;
    x2 = -in_sw;
    x3 = -in_w;
    x4 = x2 + x3;
    x5 = -in_nw;
    x6 = in_ne + x2;
    x7 = -in_e + x3;
    x8 = -in_n;
    x9 = 4 * in_se;
    x10 = -2 * in_c;
    x11 = -2 * in_e;
    x12 = 2 * in_n;
    x13 = 2 * in_w;
    x14 = -in_ne;
    x15 = in_s + in_sw + x14 + x5;

    /* Compute rho */
    /* rho = in_n + in_s + in_e + in_w + in_ne + in_nw + in_se + in_sw + in_c; */
    rho = x1;
    uX = in_e + in_ne + in_se + x4 + x5;
    uY = in_n + in_nw - in_s - in_se + x6;

    /* Compute velocity along x-axis */
    /* uX = (in_se + in_e + in_ne) - (in_sw + in_w + in_nw); */
    /* uX = uX / rho; */

    /* Compute velocity along y-axis */
    /* uY = (in_nw + in_n + in_ne) - (in_sw + in_s + in_se); */
    /* uY = uY / rho; */

    /* Impose the constantly moving upper wall */
    /* %%BOUNDARY%% */
    if (y == nY-1) {
        uX = uTop * rho;
        uY = 0.00 * rho;
    }
    /* %%BOUNDARY%% */

    /* Convert PDFs to moment space */
    /* m_0 = +1 * in_c + 1 * in_e + 1 * in_n + 1 * in_w + 1 * in_s + 1 * in_ne + 1
     * * in_nw + 1 * in_sw + 1 * in_se; */
    /* m_1 = -1 * in_c - 1 * in_e - 1 * in_n - 1 * in_w + 2 * in_s + 2 * in_ne + 2
     * * in_nw + 2 * in_sw - 4 * in_se; */
    /* m_2 = -2 * in_c - 2 * in_e - 2 * in_n - 2 * in_w + 1 * in_s + 1 * in_ne + 1
     * * in_nw + 1 * in_sw + 4 * in_se; */
    /* m_3 = +1 * in_c + 0 * in_e - 1 * in_n + 0 * in_w + 1 * in_s - 1 * in_ne - 1
     * * in_nw + 1 * in_sw + 0 * in_se; */
    /* m_4 = -2 * in_c + 0 * in_e + 2 * in_n + 0 * in_w + 1 * in_s - 1 * in_ne - 1
     * * in_nw + 1 * in_sw + 0 * in_se; */
    /* m_5 = +0 * in_c + 1 * in_e + 0 * in_n - 1 * in_w + 1 * in_s + 1 * in_ne - 1
     * * in_nw - 1 * in_sw + 0 * in_se; */
    /* m_6 = +0 * in_c - 2 * in_e + 0 * in_n + 2 * in_w + 1 * in_s + 1 * in_ne - 1
     * * in_nw - 1 * in_sw + 0 * in_se; */
    /* m_7 = +1 * in_c - 1 * in_e + 1 * in_n - 1 * in_w + 0 * in_s + 0 * in_ne + 0
     * * in_nw + 0 * in_sw + 0 * in_se; */
    /* m_8 = +0 * in_c + 0 * in_e + 0 * in_n + 0 * in_w + 1 * in_s - 1 * in_ne + 1
     * * in_nw - 1 * in_sw + 0 * in_se; */

    m_0 = x1;
    m_1 = -in_c + 2 * in_ne + 2 * in_nw + 2 * in_s + 2 * in_sw + x7 + x8 - x9;
    m_2 = in_ne + x0 + x10 + x11 - x12 - x13 + x9;
    m_3 = in_c + x15 + x8;
    m_4 = x10 + x12 + x15;
    m_5 = in_e + in_ne + in_s + x4 + x5;
    m_6 = in_s + x11 + x13 + x5 + x6;
    m_7 = in_c + in_n + x7;
    m_8 = in_nw + in_s + x14 + x2;

    /* Calculate equilibrium moments */
    x0 = uX * uX;
    x1 = 3.0 * x0;
    x2 = uY * uY;
    x3 = 3.0 * x2;

    m_eq0 = rho;
    m_eq1 = -2.0 * rho + x1 + x3;
    m_eq2 = rho - x1 - x3;
    m_eq3 = uX;
    m_eq4 = -uX;
    m_eq5 = uY;
    m_eq6 = -uY;
    m_eq7 = x0 - x2;
    m_eq8 = uX * uY;

    /* Calculate difference, relax in momentum space and collide
     * (m - s * (m - m_eq)) */
    m_eq0 = m_0 - s_0 * (m_0 - m_eq0);
    m_eq1 = m_1 - s_1 * (m_1 - m_eq1);
    m_eq2 = m_2 - s_2 * (m_2 - m_eq2);
    m_eq3 = m_3 - s_3 * (m_3 - m_eq3);
    m_eq4 = m_4 - s_4 * (m_4 - m_eq4);
    m_eq5 = m_5 - s_5 * (m_5 - m_eq5);
    m_eq6 = m_6 - s_6 * (m_6 - m_eq6);
    m_eq7 = m_7 - s_7 * (m_7 - m_eq7);
    m_eq8 = m_8 - s_8 * (m_8 - m_eq8);

    /* Inverse transform from moment space */
    x0 = 0.111111111111111 * m_eq0;
    x1 = 0.166666666666667 * m_eq3;
    x2 = 0.166666666666667 * m_eq4;
    x3 = -0.0277777777777778 * m_eq1;
    x4 = -0.0555555555555556 * m_eq2;
    x5 = 0.25 * m_eq7;
    x6 = x0 + x3 + x4 + x5;
    x7 = 0.166666666666667 * m_eq5;
    x8 = 0.166666666666667 * m_eq6;
    x9 = x0 + x3 + x4 - x5;
    x10 = -x1;
    x11 = -x7;
    x12 = 0.0833333333333333 * m_eq6;
    x13 = 0.25 * m_eq8;
    x14 = 0.0555555555555556 * m_eq1;
    x15 = 0.0277777777777778 * m_eq2;
    x16 = 0.0833333333333333 * m_eq4;
    x17 = x0 + x1 + x14 + x15 + x16;

    *out_c = -0.111111111111111 * m_eq1 + 0.111111111111111 * m_eq2 + x0;
    *out_e = x1 - x2 + x6;
    *out_n = x7 - x8 + x9;
    *out_w = x10 + x2 + x6;
    *out_s = x11 + x8 + x9;
    *out_ne = x12 + x13 + x17 + x7;
    *out_nw = x0 + x10 + x12 + x14 + x15 - x16 - x13 + x7;
    *out_sw = x0 + x10 + x11 + x13 + x14 + x15 - x16 - x12;
    *out_se = x11 + x17 - x13 - x12;
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

/* icc -O3 -fp-model precise -restrict -DDEBUG -DTIME mrt_ldc_d2q9.c -o
 * mrt_ldc_d2q9.elf */
