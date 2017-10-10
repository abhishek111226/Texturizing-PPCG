/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _RNN_H
# define _RNN_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

# if !defined(NT) && !defined(NP) && !defined(NQ) && !defined(NS)  && !defined(BT) 
/* Define sample dataset sizes. */
#  ifdef MINI_DATASET
#   define BT 1
#   define NT 20
#   define NP 20
#   define NQ 25
#   define NS 30
#  endif 

#  ifdef SMALL_DATASET
#   define BT 2
#   define NT 60
#   define NP 80
#   define NQ 100
#   define NS 110
#  endif 

#  ifdef MEDIUM_DATASET
#   define BT 3
#   define NT 100
#   define NP 200
#   define NQ 250
#   define NS 310
#  endif 

#  ifdef LARGE_DATASET
#   define BT 4
#   define NT 500
#   define NP 500
#   define NQ 650
#   define NS 700
#  endif 

#  ifdef EXTRALARGE_DATASET
#   define BT 5
#   define NT 1000
#   define NP 1000
#   define NQ 1500
#   define NS 2000
#  endif 


#endif /* !(NT NP NQ NS BT) */

# define _PB_NT POLYBENCH_LOOP_BOUND(NT,nt)
# define _PB_NP POLYBENCH_LOOP_BOUND(NP,np)
# define _PB_NQ POLYBENCH_LOOP_BOUND(NQ,nq)
# define _PB_NS POLYBENCH_LOOP_BOUND(NS,ns)
# define _PB_BT POLYBENCH_LOOP_BOUND(BT,bt)


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

#endif /* !_RNN_H */

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
