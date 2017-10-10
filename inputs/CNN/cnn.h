/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _CNN_H
# define _CNN_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

# if !defined(NN) && !defined(NK) && !defined(NP) && !defined(NQ) && !defined(NC) && !defined(NR) && !defined(NS) && !defined(NW) && !defined(NH) 
/* Define sample dataset sizes. */
#  ifdef MINI_DATASET
#   define NU 2
#   define NV 2
#   define NR 3
#   define NS 3
#   define NH 10
#   define NW 10
#   define NC 10
#   define NK 7
#   define NN 5
#  endif 

#  ifdef SMALL_DATASET
#   define NU 3
#   define NV 3
#   define NR 4
#   define NS 4
#   define NH 20
#   define NW 20
#   define NC 20
#   define NK 15
#   define NN 10
#  endif 

#  ifdef MEDIUM_DATASET
#   define NU 4
#   define NV 4
#   define NR 5
#   define NS 5
#   define NH 40
#   define NW 40
#   define NC 50
#   define NK 25
#   define NN 25
#  endif 

#  ifdef LARGE_DATASET
#   define NU 5
#   define NV 5
#   define NR 6
#   define NS 6
#   define NH 50
#   define NW 50
#   define NC 75
#   define NK 40
#   define NN 50
#  endif 

#  ifdef EXTRALARGE_DATASET
#   define NU 5
#   define NV 5
#   define NR 6
#   define NS 6
#   define NH 75
#   define NW 75
#   define NC 100
#   define NK 40
#   define NN 100
#  endif 


#endif /* !(NU NV NR NS NH NW NC NK NN) */

#   define NP (NH - NR)/NU + 1 
#   define NQ (NW - NS)/NV + 1 

# define _PB_NN POLYBENCH_LOOP_BOUND(NN,nn)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)
# define _PB_NP POLYBENCH_LOOP_BOUND(NP,np)
# define _PB_NQ POLYBENCH_LOOP_BOUND(NQ,nq)
# define _PB_NC POLYBENCH_LOOP_BOUND(NC,nc)
# define _PB_NR POLYBENCH_LOOP_BOUND(NR,nr)
# define _PB_NS POLYBENCH_LOOP_BOUND(NS,ns)
# define _PB_NH POLYBENCH_LOOP_BOUND(NH,nh)
# define _PB_NW POLYBENCH_LOOP_BOUND(NW,nw)


/* Default data type */
# if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#  define DATA_TYPE_IS_DOUBLE
# endif

#ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
#endif 

#ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
# endif

#ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
# endif

#endif /* !_GEMM_H */

