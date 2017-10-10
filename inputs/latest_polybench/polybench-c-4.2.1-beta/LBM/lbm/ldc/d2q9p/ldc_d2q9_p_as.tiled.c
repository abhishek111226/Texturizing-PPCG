#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/************************************************************************/
/* Lid Driven Cavity(LDC) - 2D simulation using LBM with a D2Q9 lattice */
/*                                                                      */
/* Scheme: 2-Grid, Pull, Compute, Keep                                  */
/*                                                                      */
/* Lattice naming convention                                            */
/*   c6[NW]  c2[N]  c5[NE]                                              */
/*   c3[W]   c0[C]  c1[E]                                               */
/*   c7[SW]  c4[S]  c8[SE]                                              */
/*                                                                      */
/* Author: Irshad Pananilath <pmirshad+code@gmail.com>                  */
/*                                                                      */
/************************************************************************/

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
#define nTimesteps 50000  // Number of timesteps for the simulation
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
const double re = 100;
const double uTop = 0.01;

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
    omega = 2.0 / ((6.0 * uTop * (nX - 0.5) / re) + 1.0);
    one_minus_omega = 1.0 - omega;

    printf(
        "2D Lid Driven Cavity simulation with D2Q9 lattice:\n"
        "\tfilename   : %s\n"
        "\tscheme     : 2-Grid, Fused, Pull\n"
        "\tgrid size  : %d x %d = %.2lf * 10^3 Cells\n"
        "\tnTimeSteps : %d\n"
        "\tRe         : %.2lf\n"
        "\tuTop       : %.6lf\n"
        "\tomega      : %.6lf\n",
        __FILE__, nX, nY, nX * nY / 1.0e3, nTimesteps, re, uTop, omega);

    /* Initialize all 9 PDFs for each point in the domain to 1.0 */
    for (y = 0; y < nY; y++) {
        for (x = 0; x < nX; x++) {
            for (k = 0; k < nK; k++) {
                grid[0][y][x][k] = 1.0;
                grid[1][y][x][k] = 1.0;
            }
        }
    }

    /* To satisfy PET */
    short _nX = nX;
    short _nY = nY;
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
    register int lbv, ubv;
    /* Start of CLooG code */
    if ((_nTimesteps >= 1) && (_nX >= 1) && (_nY >= 1)) {
        for (t1=0; t1<=floord(_nTimesteps-1,32); t1++) {
            for (t2=t1; t2<=min(floord(2*_nTimesteps+_nY-2,64),floord(64*t1+_nY+62,64)); t2++) {
                for (t3=t1; t3<=min(floord(2*_nTimesteps+_nX-2,64),floord(64*t1+_nX+62,64)); t3++) {
                    if ((_nX == 1) && (_nY == 1) && (t1 == t2) && (t1 == t3)) {
                        for (t4=32*t1; t4<=min(_nTimesteps-1,32*t1+31); t4++) {
                            lbm_kernel(grid[ t4 % 2][ 0][ 0][0], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0][2], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0][4], grid[ t4 % 2][ 0][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ t4 % 2][ 0][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( t4 + 1) % 2][ 0][ 0][0], &grid[( t4 + 1) % 2][ 0][ 0][2], &grid[( t4 + 1) % 2][ 0][ 0][4], &grid[( t4 + 1) % 2][ 0][ 0][1], &grid[( t4 + 1) % 2][ 0][ 0][3], &grid[( t4 + 1) % 2][ 0][ 0][5], &grid[( t4 + 1) % 2][ 0][ 0][6], &grid[( t4 + 1) % 2][ 0][ 0][8], &grid[( t4 + 1) % 2][ 0][ 0][7], ( t4), ( 0), ( 0));;
                        }
                    }
                    if ((_nX >= 2) && (t1 <= floord(64*t2-_nY,64)) && (t2 <= floord(64*t3+_nY+60,64)) && (t2 >= ceild(64*t3+_nY-_nX+1,64))) {
                        if (_nY%2 == 0) {
                            lbv=max(ceild(64*t2-_nY,2),32*t3);
                            ubv=min(floord(64*t2-_nY+_nX-1,2),32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][0], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ ((-64*t2+2*t7+_nY)/2)][2], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ ((-64*t2+2*t7+_nY)/2)][4], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2) == 0 ? _nX - 1 : ((-64*t2+2*t7+_nY)/2) - 1][1], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2) + 1 == _nX ? 0 : ((-64*t2+2*t7+_nY)/2) + 1][3], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ ((-64*t2+2*t7+_nY)/2) == 0 ? _nX - 1 : ((-64*t2+2*t7+_nY)/2) - 1][5], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ ((-64*t2+2*t7+_nY)/2) + 1 == _nX ? 0 : ((-64*t2+2*t7+_nY)/2) + 1][6], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ ((-64*t2+2*t7+_nY)/2) == 0 ? _nX - 1 : ((-64*t2+2*t7+_nY)/2) - 1][8], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ ((-64*t2+2*t7+_nY)/2) + 1 == _nX ? 0 : ((-64*t2+2*t7+_nY)/2) + 1][7], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][0], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][2], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][4], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][1], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][3], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][5], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][6], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][8], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((-64*t2+2*t7+_nY)/2)][7], ( ((64*t2-_nY)/2)), ( (_nY/2)), ( ((-64*t2+2*t7+_nY)/2)));;
                            }
                            lbv=max(ceild(64*t2-_nY+2,2),32*t3);
                            ubv=min(floord(64*t2-_nY+_nX,2),32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][0], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ ((64*t2-2*t7-_nY+2*_nX)/2)][2], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ ((64*t2-2*t7-_nY+2*_nX)/2)][4], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2) == 0 ? _nX - 1 : ((64*t2-2*t7-_nY+2*_nX)/2) - 1][1], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2) + 1 == _nX ? 0 : ((64*t2-2*t7-_nY+2*_nX)/2) + 1][3], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ ((64*t2-2*t7-_nY+2*_nX)/2) == 0 ? _nX - 1 : ((64*t2-2*t7-_nY+2*_nX)/2) - 1][5], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ ((64*t2-2*t7-_nY+2*_nX)/2) + 1 == _nX ? 0 : ((64*t2-2*t7-_nY+2*_nX)/2) + 1][6], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ ((64*t2-2*t7-_nY+2*_nX)/2) == 0 ? _nX - 1 : ((64*t2-2*t7-_nY+2*_nX)/2) - 1][8], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ ((64*t2-2*t7-_nY+2*_nX)/2) + 1 == _nX ? 0 : ((64*t2-2*t7-_nY+2*_nX)/2) + 1][7], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][0], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][2], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][4], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][1], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][3], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][5], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][6], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][8], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ ((64*t2-2*t7-_nY+2*_nX)/2)][7], ( ((64*t2-_nY)/2)), ( (_nY/2)), ( ((64*t2-2*t7-_nY+2*_nX)/2)));;
                            }
                        }
                    }
                    if ((_nX >= 2) && (t1 == t3) && (64*t1 == 64*t2-_nY-62)) {
                        if (_nY%2 == 0) {
                            if ((_nY+62)%64 == 0) {
                                lbm_kernel(grid[ (32*t1+31) % 2][ (_nY/2)][ 0][0], grid[ (32*t1+31) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ 0][2], grid[ (32*t1+31) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ 0][4], grid[ (32*t1+31) % 2][ (_nY/2)][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ (32*t1+31) % 2][ (_nY/2)][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ (32*t1+31) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ (32*t1+31) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ (32*t1+31) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ (32*t1+31) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][0], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][2], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][4], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][1], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][3], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][5], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][6], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][8], &grid[( (32*t1+31) + 1) % 2][ (_nY/2)][ 0][7], ( (32*t1+31)), ( (_nY/2)), ( 0));;
                            }
                        }
                    }
                    if ((_nX == 1) && (t1 == t3) && (t1 <= floord(64*t2-_nY,64))) {
                        if (_nY%2 == 0) {
                            lbm_kernel(grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ 0][0], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ 0][2], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ 0][4], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][0], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][2], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][4], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][1], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][3], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][5], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][6], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][8], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ 0][7], ( ((64*t2-_nY)/2)), ( (_nY/2)), ( 0));;
                        }
                    }
                    if ((_nY >= 2) && (t1 <= floord(64*t3-_nX,64)) && (t2 <= floord(64*t3+_nY-_nX-1,64)) && (t2 >= ceild(64*t3-_nX-60,64))) {
                        if (_nX%2 == 0) {
                            for (t6=max(ceild(64*t3-_nX,2),32*t2); t6<=min(floord(64*t3+_nY-_nX-1,2),32*t2+31); t6++) {
                                lbm_kernel(grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][0], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2) == 0 ? _nY - 1 : ((-64*t3+2*t6+_nX)/2) - 1][ (_nX/2)][2], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2) + 1 == _nY ? 0 : ((-64*t3+2*t6+_nX)/2) + 1][ (_nX/2)][4], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][1], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][3], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2) == 0 ? _nY - 1 : ((-64*t3+2*t6+_nX)/2) - 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][5], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2) == 0 ? _nY - 1 : ((-64*t3+2*t6+_nX)/2) - 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][6], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2) + 1 == _nY ? 0 : ((-64*t3+2*t6+_nX)/2) + 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][8], grid[ ((64*t3-_nX)/2) % 2][ ((-64*t3+2*t6+_nX)/2) + 1 == _nY ? 0 : ((-64*t3+2*t6+_nX)/2) + 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][7], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][0], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][2], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][4], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][1], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][3], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][5], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][6], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][8], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((-64*t3+2*t6+_nX)/2)][ (_nX/2)][7], ( ((64*t3-_nX)/2)), ( ((-64*t3+2*t6+_nX)/2)), ( (_nX/2)));;
                            }
                            for (t6=max(ceild(64*t3-_nX+2,2),32*t2); t6<=min(floord(64*t3+_nY-_nX,2),32*t2+31); t6++) {
                                lbm_kernel(grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][0], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2) == 0 ? _nY - 1 : ((64*t3-2*t6+2*_nY-_nX)/2) - 1][ (_nX/2)][2], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2) + 1 == _nY ? 0 : ((64*t3-2*t6+2*_nY-_nX)/2) + 1][ (_nX/2)][4], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][1], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][3], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2) == 0 ? _nY - 1 : ((64*t3-2*t6+2*_nY-_nX)/2) - 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][5], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2) == 0 ? _nY - 1 : ((64*t3-2*t6+2*_nY-_nX)/2) - 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][6], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2) + 1 == _nY ? 0 : ((64*t3-2*t6+2*_nY-_nX)/2) + 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][8], grid[ ((64*t3-_nX)/2) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2) + 1 == _nY ? 0 : ((64*t3-2*t6+2*_nY-_nX)/2) + 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][7], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][0], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][2], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][4], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][1], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][3], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][5], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][6], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][8], &grid[( ((64*t3-_nX)/2) + 1) % 2][ ((64*t3-2*t6+2*_nY-_nX)/2)][ (_nX/2)][7], ( ((64*t3-_nX)/2)), ( ((64*t3-2*t6+2*_nY-_nX)/2)), ( (_nX/2)));;
                            }
                        }
                    }
                    if ((_nY >= 2) && (t1 == t2) && (64*t1 == 64*t3-_nX-62)) {
                        if (_nX%2 == 0) {
                            if ((_nX+62)%64 == 0) {
                                lbm_kernel(grid[ (32*t1+31) % 2][ 0][ (_nX/2)][0], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (_nX/2)][2], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (_nX/2)][4], grid[ (32*t1+31) % 2][ 0][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][1], grid[ (32*t1+31) % 2][ 0][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][3], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][5], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][6], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][8], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][7], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][0], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][2], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][4], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][1], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][3], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][5], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][6], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][8], &grid[( (32*t1+31) + 1) % 2][ 0][ (_nX/2)][7], ( (32*t1+31)), ( 0), ( (_nX/2)));;
                            }
                        }
                    }
                    if ((_nY == 1) && (t1 == t2) && (t1 <= floord(64*t3-_nX,64))) {
                        if (_nX%2 == 0) {
                            lbm_kernel(grid[ ((64*t3-_nX)/2) % 2][ 0][ (_nX/2)][0], grid[ ((64*t3-_nX)/2) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (_nX/2)][2], grid[ ((64*t3-_nX)/2) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (_nX/2)][4], grid[ ((64*t3-_nX)/2) % 2][ 0][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][1], grid[ ((64*t3-_nX)/2) % 2][ 0][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][3], grid[ ((64*t3-_nX)/2) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][5], grid[ ((64*t3-_nX)/2) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][6], grid[ ((64*t3-_nX)/2) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][8], grid[ ((64*t3-_nX)/2) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][7], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][0], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][2], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][4], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][1], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][3], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][5], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][6], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][8], &grid[( ((64*t3-_nX)/2) + 1) % 2][ 0][ (_nX/2)][7], ( ((64*t3-_nX)/2)), ( 0), ( (_nX/2)));;
                        }
                    }
                    if ((t1 <= floord(64*t2-_nY,64)) && (64*t2 == 64*t3+_nY-_nX)) {
                        if (_nY%2 == 0) {
                            if (_nX%2 == 0) {
                                if ((63*_nY+_nX)%64 == 0) {
                                    lbm_kernel(grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ (_nX/2)][0], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ (_nX/2)][2], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ (_nX/2)][4], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][1], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2)][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][3], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][5], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) == 0 ? _nY - 1 : (_nY/2) - 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][6], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ (_nX/2) == 0 ? _nX - 1 : (_nX/2) - 1][8], grid[ ((64*t2-_nY)/2) % 2][ (_nY/2) + 1 == _nY ? 0 : (_nY/2) + 1][ (_nX/2) + 1 == _nX ? 0 : (_nX/2) + 1][7], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][0], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][2], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][4], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][1], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][3], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][5], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][6], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][8], &grid[( ((64*t2-_nY)/2) + 1) % 2][ (_nY/2)][ (_nX/2)][7], ( ((64*t2-_nY)/2)), ( (_nY/2)), ( (_nX/2)));;
                                }
                            }
                        }
                    }
                    if ((_nX >= 2) && (_nY >= 2)) {
                        for (t4=max(max(ceild(64*t2-_nY+1,2),ceild(64*t3-_nX+1,2)),32*t1); t4<=min(min(min(_nTimesteps-1,32*t1+31),32*t2+30),32*t3+30); t4++) {
                            for (t6=max(32*t2,t4); t6<=min(floord(2*t4+_nY-1,2),32*t2+31); t6++) {
                                lbv=max(32*t3,t4);
                                ubv=min(floord(2*t4+_nX-1,2),32*t3+31);
#pragma ivdep
#pragma vector always
                                for (t7=lbv; t7<=ubv; t7++) {
                                    lbm_kernel(grid[ t4 % 2][ (-t4+t6)][ (-t4+t7)][0], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ (-t4+t7)][2], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ (-t4+t7)][4], grid[ t4 % 2][ (-t4+t6)][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][1], grid[ t4 % 2][ (-t4+t6)][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][3], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][5], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][6], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][8], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][7], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][0], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][2], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][4], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][1], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][3], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][5], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][6], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][8], &grid[( t4 + 1) % 2][ (-t4+t6)][ (-t4+t7)][7], ( t4), ( (-t4+t6)), ( (-t4+t7)));;
                                }
                            }
                            for (t6=max(32*t2,t4+1); t6<=min(floord(2*t4+_nY,2),32*t2+31); t6++) {
                                lbv=max(32*t3,t4);
                                ubv=min(floord(2*t4+_nX-1,2),32*t3+31);
#pragma ivdep
#pragma vector always
                                for (t7=lbv; t7<=ubv; t7++) {
                                    lbm_kernel(grid[ t4 % 2][ (t4-t6+_nY)][ (-t4+t7)][0], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ (-t4+t7)][2], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ (-t4+t7)][4], grid[ t4 % 2][ (t4-t6+_nY)][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][1], grid[ t4 % 2][ (t4-t6+_nY)][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][3], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][5], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][6], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][8], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][7], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][0], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][2], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][4], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][1], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][3], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][5], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][6], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][8], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (-t4+t7)][7], ( t4), ( (t4-t6+_nY)), ( (-t4+t7)));;
                                }
                            }
                            for (t6=max(32*t2,t4); t6<=min(floord(2*t4+_nY-1,2),32*t2+31); t6++) {
                                lbv=max(32*t3,t4+1);
                                ubv=min(floord(2*t4+_nX,2),32*t3+31);
#pragma ivdep
#pragma vector always
                                for (t7=lbv; t7<=ubv; t7++) {
                                    lbm_kernel(grid[ t4 % 2][ (-t4+t6)][ (t4-t7+_nX)][0], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ (t4-t7+_nX)][2], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ (t4-t7+_nX)][4], grid[ t4 % 2][ (-t4+t6)][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][1], grid[ t4 % 2][ (-t4+t6)][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][3], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][5], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][6], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][8], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][7], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][0], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][2], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][4], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][1], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][3], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][5], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][6], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][8], &grid[( t4 + 1) % 2][ (-t4+t6)][ (t4-t7+_nX)][7], ( t4), ( (-t4+t6)), ( (t4-t7+_nX)));;
                                }
                            }
                            for (t6=max(32*t2,t4+1); t6<=min(floord(2*t4+_nY,2),32*t2+31); t6++) {
                                lbv=max(32*t3,t4+1);
                                ubv=min(floord(2*t4+_nX,2),32*t3+31);
#pragma ivdep
#pragma vector always
                                for (t7=lbv; t7<=ubv; t7++) {
                                    lbm_kernel(grid[ t4 % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][0], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ (t4-t7+_nX)][2], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ (t4-t7+_nX)][4], grid[ t4 % 2][ (t4-t6+_nY)][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][1], grid[ t4 % 2][ (t4-t6+_nY)][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][3], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][5], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][6], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][8], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][7], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][0], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][2], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][4], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][1], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][3], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][5], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][6], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][8], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ (t4-t7+_nX)][7], ( t4), ( (t4-t6+_nY)), ( (t4-t7+_nX)));;
                                }
                            }
                        }
                    }
                    if ((_nX >= 2) && (t1 == t3) && (t1 <= min(floord(_nTimesteps-32,32),t2-1)) && (t1 >= ceild(64*t2-_nY-61,64))) {
                        for (t6=32*t2; t6<=min(floord(64*t1+_nY+61,2),32*t2+31); t6++) {
                            lbm_kernel(grid[ (32*t1+31) % 2][ (-32*t1+t6-31)][ 0][0], grid[ (32*t1+31) % 2][ (-32*t1+t6-31) == 0 ? _nY - 1 : (-32*t1+t6-31) - 1][ 0][2], grid[ (32*t1+31) % 2][ (-32*t1+t6-31) + 1 == _nY ? 0 : (-32*t1+t6-31) + 1][ 0][4], grid[ (32*t1+31) % 2][ (-32*t1+t6-31)][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ (32*t1+31) % 2][ (-32*t1+t6-31)][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ (32*t1+31) % 2][ (-32*t1+t6-31) == 0 ? _nY - 1 : (-32*t1+t6-31) - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ (32*t1+31) % 2][ (-32*t1+t6-31) == 0 ? _nY - 1 : (-32*t1+t6-31) - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ (32*t1+31) % 2][ (-32*t1+t6-31) + 1 == _nY ? 0 : (-32*t1+t6-31) + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ (32*t1+31) % 2][ (-32*t1+t6-31) + 1 == _nY ? 0 : (-32*t1+t6-31) + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][0], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][2], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][4], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][1], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][3], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][5], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][6], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][8], &grid[( (32*t1+31) + 1) % 2][ (-32*t1+t6-31)][ 0][7], ( (32*t1+31)), ( (-32*t1+t6-31)), ( 0));;
                        }
                        for (t6=32*t2; t6<=min(floord(64*t1+_nY+62,2),32*t2+31); t6++) {
                            lbm_kernel(grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31)][ 0][0], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31) == 0 ? _nY - 1 : (32*t1-t6+_nY+31) - 1][ 0][2], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31) + 1 == _nY ? 0 : (32*t1-t6+_nY+31) + 1][ 0][4], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31)][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31)][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31) == 0 ? _nY - 1 : (32*t1-t6+_nY+31) - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31) == 0 ? _nY - 1 : (32*t1-t6+_nY+31) - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31) + 1 == _nY ? 0 : (32*t1-t6+_nY+31) + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ (32*t1+31) % 2][ (32*t1-t6+_nY+31) + 1 == _nY ? 0 : (32*t1-t6+_nY+31) + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][0], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][2], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][4], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][1], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][3], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][5], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][6], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][8], &grid[( (32*t1+31) + 1) % 2][ (32*t1-t6+_nY+31)][ 0][7], ( (32*t1+31)), ( (32*t1-t6+_nY+31)), ( 0));;
                        }
                    }
                    if ((_nX == 1) && (_nY >= 2) && (t1 == t3)) {
                        for (t4=max(ceild(64*t2-_nY+1,2),32*t1); t4<=min(min(_nTimesteps-1,32*t1+31),32*t2+30); t4++) {
                            for (t6=max(32*t2,t4); t6<=min(floord(2*t4+_nY-1,2),32*t2+31); t6++) {
                                lbm_kernel(grid[ t4 % 2][ (-t4+t6)][ 0][0], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ 0][2], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ 0][4], grid[ t4 % 2][ (-t4+t6)][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ t4 % 2][ (-t4+t6)][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ t4 % 2][ (-t4+t6) == 0 ? _nY - 1 : (-t4+t6) - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ t4 % 2][ (-t4+t6) + 1 == _nY ? 0 : (-t4+t6) + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][0], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][2], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][4], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][1], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][3], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][5], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][6], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][8], &grid[( t4 + 1) % 2][ (-t4+t6)][ 0][7], ( t4), ( (-t4+t6)), ( 0));;
                            }
                            for (t6=max(32*t2,t4+1); t6<=min(floord(2*t4+_nY,2),32*t2+31); t6++) {
                                lbm_kernel(grid[ t4 % 2][ (t4-t6+_nY)][ 0][0], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ 0][2], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ 0][4], grid[ t4 % 2][ (t4-t6+_nY)][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ t4 % 2][ (t4-t6+_nY)][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ t4 % 2][ (t4-t6+_nY) == 0 ? _nY - 1 : (t4-t6+_nY) - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ t4 % 2][ (t4-t6+_nY) + 1 == _nY ? 0 : (t4-t6+_nY) + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][0], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][2], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][4], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][1], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][3], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][5], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][6], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][8], &grid[( t4 + 1) % 2][ (t4-t6+_nY)][ 0][7], ( t4), ( (t4-t6+_nY)), ( 0));;
                            }
                        }
                    }
                    if ((_nY >= 2) && (t1 == t2) && (t1 <= min(floord(_nTimesteps-32,32),t3-1)) && (t1 >= ceild(64*t3-_nX-61,64))) {
                        lbv=32*t3;
                        ubv=min(floord(64*t1+_nX+61,2),32*t3+31);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t1+31) % 2][ 0][ (-32*t1+t7-31)][0], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (-32*t1+t7-31)][2], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (-32*t1+t7-31)][4], grid[ (32*t1+31) % 2][ 0][ (-32*t1+t7-31) == 0 ? _nX - 1 : (-32*t1+t7-31) - 1][1], grid[ (32*t1+31) % 2][ 0][ (-32*t1+t7-31) + 1 == _nX ? 0 : (-32*t1+t7-31) + 1][3], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (-32*t1+t7-31) == 0 ? _nX - 1 : (-32*t1+t7-31) - 1][5], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (-32*t1+t7-31) + 1 == _nX ? 0 : (-32*t1+t7-31) + 1][6], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (-32*t1+t7-31) == 0 ? _nX - 1 : (-32*t1+t7-31) - 1][8], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (-32*t1+t7-31) + 1 == _nX ? 0 : (-32*t1+t7-31) + 1][7], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][0], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][2], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][4], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][1], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][3], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][5], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][6], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][8], &grid[( (32*t1+31) + 1) % 2][ 0][ (-32*t1+t7-31)][7], ( (32*t1+31)), ( 0), ( (-32*t1+t7-31)));;
                        }
                        lbv=32*t3;
                        ubv=min(floord(64*t1+_nX+62,2),32*t3+31);
#pragma ivdep
#pragma vector always
                        for (t7=lbv; t7<=ubv; t7++) {
                            lbm_kernel(grid[ (32*t1+31) % 2][ 0][ (32*t1-t7+_nX+31)][0], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (32*t1-t7+_nX+31)][2], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (32*t1-t7+_nX+31)][4], grid[ (32*t1+31) % 2][ 0][ (32*t1-t7+_nX+31) == 0 ? _nX - 1 : (32*t1-t7+_nX+31) - 1][1], grid[ (32*t1+31) % 2][ 0][ (32*t1-t7+_nX+31) + 1 == _nX ? 0 : (32*t1-t7+_nX+31) + 1][3], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (32*t1-t7+_nX+31) == 0 ? _nX - 1 : (32*t1-t7+_nX+31) - 1][5], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (32*t1-t7+_nX+31) + 1 == _nX ? 0 : (32*t1-t7+_nX+31) + 1][6], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (32*t1-t7+_nX+31) == 0 ? _nX - 1 : (32*t1-t7+_nX+31) - 1][8], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (32*t1-t7+_nX+31) + 1 == _nX ? 0 : (32*t1-t7+_nX+31) + 1][7], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][0], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][2], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][4], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][1], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][3], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][5], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][6], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][8], &grid[( (32*t1+31) + 1) % 2][ 0][ (32*t1-t7+_nX+31)][7], ( (32*t1+31)), ( 0), ( (32*t1-t7+_nX+31)));;
                        }
                    }
                    if ((_nY >= 2) && (t1 == t2) && (t1 == t3) && (t1 <= floord(_nTimesteps-32,32))) {
                        lbm_kernel(grid[ (32*t1+31) % 2][ 0][ 0][0], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0][2], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0][4], grid[ (32*t1+31) % 2][ 0][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ (32*t1+31) % 2][ 0][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][0], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][2], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][4], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][1], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][3], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][5], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][6], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][8], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][7], ( (32*t1+31)), ( 0), ( 0));;
                    }
                    if ((_nX >= 2) && (_nY == 1) && (t1 == t2)) {
                        for (t4=max(ceild(64*t3-_nX+1,2),32*t1); t4<=min(min(_nTimesteps-1,32*t1+31),32*t3+30); t4++) {
                            lbv=max(32*t3,t4);
                            ubv=min(floord(2*t4+_nX-1,2),32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ 0][ (-t4+t7)][0], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (-t4+t7)][2], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (-t4+t7)][4], grid[ t4 % 2][ 0][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][1], grid[ t4 % 2][ 0][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][3], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][5], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][6], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (-t4+t7) == 0 ? _nX - 1 : (-t4+t7) - 1][8], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (-t4+t7) + 1 == _nX ? 0 : (-t4+t7) + 1][7], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][0], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][2], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][4], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][1], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][3], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][5], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][6], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][8], &grid[( t4 + 1) % 2][ 0][ (-t4+t7)][7], ( t4), ( 0), ( (-t4+t7)));;
                            }
                            lbv=max(32*t3,t4+1);
                            ubv=min(floord(2*t4+_nX,2),32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t7=lbv; t7<=ubv; t7++) {
                                lbm_kernel(grid[ t4 % 2][ 0][ (t4-t7+_nX)][0], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (t4-t7+_nX)][2], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (t4-t7+_nX)][4], grid[ t4 % 2][ 0][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][1], grid[ t4 % 2][ 0][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][3], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][5], grid[ t4 % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][6], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (t4-t7+_nX) == 0 ? _nX - 1 : (t4-t7+_nX) - 1][8], grid[ t4 % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ (t4-t7+_nX) + 1 == _nX ? 0 : (t4-t7+_nX) + 1][7], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][0], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][2], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][4], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][1], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][3], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][5], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][6], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][8], &grid[( t4 + 1) % 2][ 0][ (t4-t7+_nX)][7], ( t4), ( 0), ( (t4-t7+_nX)));;
                            }
                        }
                    }
                    if ((_nX >= 2) && (_nY == 1) && (t1 == t2) && (t1 == t3) && (t1 <= floord(_nTimesteps-32,32))) {
                        lbm_kernel(grid[ (32*t1+31) % 2][ 0][ 0][0], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0][2], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0][4], grid[ (32*t1+31) % 2][ 0][ 0 == 0 ? _nX - 1 : 0 - 1][1], grid[ (32*t1+31) % 2][ 0][ 0 + 1 == _nX ? 0 : 0 + 1][3], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0 == 0 ? _nX - 1 : 0 - 1][5], grid[ (32*t1+31) % 2][ 0 == 0 ? _nY - 1 : 0 - 1][ 0 + 1 == _nX ? 0 : 0 + 1][6], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0 == 0 ? _nX - 1 : 0 - 1][8], grid[ (32*t1+31) % 2][ 0 + 1 == _nY ? 0 : 0 + 1][ 0 + 1 == _nX ? 0 : 0 + 1][7], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][0], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][2], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][4], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][1], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][3], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][5], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][6], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][8], &grid[( (32*t1+31) + 1) % 2][ 0][ 0][7], ( (32*t1+31)), ( 0), ( 0));;
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
    double lbm_sum_e, lbm_sum_w, lbm_sum_n, lbm_sum_s, lbm_rho, lbm_inv_rho;
    double lbm_uX, lbm_uY, lbm_uX2, lbm_uY2, lbm_u2, lbm_one_minus_u2;
    double lbm_omega_w1_rho, lbm_omega_w2_rho, lbm_omega_w3_rho;
    double lbm_30_uY, lbm_45_uY2, lbm_30_uX, lbm_45_uX2;

    double lbm_out_c, lbm_out_n, lbm_out_s, lbm_out_e, lbm_out_w, lbm_out_ne,
           lbm_out_nw, lbm_out_se, lbm_out_sw;
    double lbm_uX_plus_uY, lbm_uX_minus_uY, lbm_30_uX_plus_uY, lbm_30_uX_minus_uY,
           lbm_45_uX_plus_uY, lbm_45_uX_minus_uY;
    /* %%INIT%% */

    /* #<{(| %%BOUNDARY%% |)}># */
    /* if (((y == 0 || y == nY - 1)) || */
    /*     ((x == 0 || x == nX - 1) ) { */
    /*   #<{(| Bounce back boundary condition at walls |)}># */
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

    /* Algebraic simplification */

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

    /* %%BOUNDARY%% */
    /* Impose the constantly moving upper wall */
    if (y == nY -1) {
        lbm_uX = uTop;
        lbm_uY = 0.00;
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

    lbm_out_n = one_minus_omega * in_n + lbm_omega_w2_rho * (lbm_one_minus_u2 + lbm_30_uY + lbm_45_uY2);
    lbm_out_s = one_minus_omega * in_s + lbm_omega_w2_rho * (lbm_one_minus_u2 - lbm_30_uY + lbm_45_uY2);
    lbm_out_e = one_minus_omega * in_e + lbm_omega_w2_rho * (lbm_one_minus_u2 + lbm_30_uX + lbm_45_uX2);
    lbm_out_w = one_minus_omega * in_w + lbm_omega_w2_rho * (lbm_one_minus_u2 - lbm_30_uX + lbm_45_uX2);

    lbm_out_ne = one_minus_omega * in_ne + lbm_omega_w3_rho * (lbm_one_minus_u2 + lbm_30_uX_plus_uY + lbm_45_uX_plus_uY);
    lbm_out_nw = one_minus_omega * in_nw + lbm_omega_w3_rho * (lbm_one_minus_u2 - lbm_30_uX_minus_uY + lbm_45_uX_minus_uY);
    lbm_out_se = one_minus_omega * in_se + lbm_omega_w3_rho * (lbm_one_minus_u2 + lbm_30_uX_minus_uY + lbm_45_uX_minus_uY);
    lbm_out_sw = one_minus_omega * in_sw + lbm_omega_w3_rho * (lbm_one_minus_u2 - lbm_30_uX_plus_uY + lbm_45_uX_plus_uY);

    *out_c = lbm_out_c;

    *out_n = lbm_out_n;
    *out_s = lbm_out_s;
    *out_e = lbm_out_e;
    *out_w = lbm_out_w;

    *out_ne = lbm_out_ne;
    *out_nw = lbm_out_nw;
    *out_se = lbm_out_se;
    *out_sw = lbm_out_sw;
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

/* icc -O3 -fp-model precise -restrict -DDEBUG -DTIME ldc_d2q9.c -o ldc_d2q9.elf */
