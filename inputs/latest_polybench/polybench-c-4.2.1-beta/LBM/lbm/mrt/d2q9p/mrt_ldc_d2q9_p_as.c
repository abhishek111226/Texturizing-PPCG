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
//#include <omp.h>

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

#pragma scop
    /* Kernel of the code */
    for (t = 0; t < _nTimesteps; t++) {
#pragma omp parallel for private(x)
    for (y = 0; y < 2*_nY; y++) {
#pragma ivdep
      for (x = 0; x < 2*_nX; x++) {
        lbm_kernel(grid[t % 2][y][x][C],
                   grid[t % 2][y==0?2*_nY-1:y - 1][x + 0][N],
                   grid[t % 2][y==2*_nY-1?0:y + 1][x + 0][S],
                   grid[t % 2][y + 0][x==0?2*_nX-1:x - 1][E],
                   grid[t % 2][y + 0][x==2*_nX-1?0:x + 1][W],
                   grid[t % 2][y==0?2*_nY-1:y - 1][x==0?2*_nX-1:x - 1][NE],
                   grid[t % 2][y==0?2*_nY-1:y-1][x==2*_nX-1?0:x+1][NW],
                   grid[t % 2][y==2*_nY-1?0:y+1][x==0?2*_nX-1:x-1][SE],
                   grid[t % 2][y==2*_nY-1?0:y+1][x==2*_nX-1?0:x+1][SW],
                   &grid[(t + 1) % 2][y][x][C],
                   &grid[(t + 1) % 2][y][x][N],
                   &grid[(t + 1) % 2][y][x][S],
                   &grid[(t + 1) % 2][y][x][E],
                   &grid[(t + 1) % 2][y][x][W],
                   &grid[(t + 1) % 2][y][x][NE],
                   &grid[(t + 1) % 2][y][x][NW],
                   &grid[(t + 1) % 2][y][x][SE],
                   &grid[(t + 1) % 2][y][x][SW],
                        t, y, x);
            }
        }
        /* #<{(| East Boundary  |)}># */
        /* for (y=0;y < _nY;y++){ */
        /*     grid[(t+1)%2][y][0][E]=grid[(t+1)%2][y][nX][E]; */
        /*     grid[(t+1)%2][y][0][NE]=grid[(t+1)%2][y][nX][NE]; */
        /*     grid[(t+1)%2][y][0][SE]=grid[(t+1)%2][y][nX][SE]; */
        /* }    */
        /* #<{(| West Boundary |)}># */
        /* for (y=0;y < _nY;y++){ */
        /*     grid[(t+1)%2][y][_nX-1][W]=grid[(t+1)%2][y][1][W]; */
        /*     grid[(t+1)%2][y][_nX-1][NW]=grid[(t+1)%2][y][1][NW]; */
        /*     grid[(t+1)%2][y][_nX-1][SW]=grid[(t+1)%2][y][1][SW]; */
        /* }    */
        /* #<{(| North Boundary |)}># */
        /* for (x=0;x < _nX;x++){ */
        /*     grid[(t+1)%2][0][x][N]=grid[(t+1)%2][nY][x][N]; */
        /*     grid[(t+1)%2][0][x][NE]=grid[(t+1)%2][nY][x][NE]; */
        /*     grid[(t+1)%2][0][x][NW]=grid[(t+1)%2][nY][x][NW]; */
        /* }    */
        /* #<{(| South Boundary |)}># */
        /* for (x=0;x < _nX;x++){ */
        /*     grid[(t+1)%2][_nY-1][x][S]=grid[(t+1)%2][1][x][S]; */
        /*     grid[(t+1)%2][_nY-1][x][SE]=grid[(t+1)%2][1][x][SE]; */
        /*     grid[(t+1)%2][_nY-1][x][SW]=grid[(t+1)%2][1][x][SW]; */
        /* }    */
        /* grid[(t+1)%2][0][0][NE]=grid[(t+1)%2][nY][nX][NE]; */
        /* grid[(t+1)%2][_nX-1][_nX-1][SE]=grid[(t+1)%2][1][1][SE]; */
        /* grid[(t+1)%2][0][_nX-1][NW]=grid[(t+1)%2][nY][0][NW]; */
        /* grid[(t+1)%2][_nX-1][0][SE]=grid[(t+1)%2][nY][0][SE]; */
        /*  */
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
