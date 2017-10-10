#include <stdio.h> 
#define DEVICECODE true 
#include "lennardJones_kernel.hu"
__global__ void kernel0(double cutforcesq, double epsilon, float f[8400][3], double sigma6)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    double private_fix;
    double private_fiy;
    double private_fiz;
    double private_xtmp;
    double private_ytmp;
    double private_ztmp;
    double private_delx;
    double private_dely;
    double private_delz;
    double private_rsq;
    double private_sr2;
    double private_sr6;
    double private_force;

    if (32 * b0 + t0 <= 8399) {
      private_xtmp = (tex2D(texRef_x, 0, 32 * b0 + t0));
      private_fix = 0;
      private_ztmp = (tex2D(texRef_x, 2, 32 * b0 + t0));
      private_ytmp = (tex2D(texRef_x, 1, 32 * b0 + t0));
      private_fiz = 0;
      private_fiy = 0;
      for (int c2 = 0; c2 < 32 * b0 + t0; c2 += 1) {
        private_delx = (private_xtmp - (tex2D(texRef_x, 0, c2)));
        private_dely = (private_ytmp - (tex2D(texRef_x, 1, c2)));
        private_delz = (private_ztmp - (tex2D(texRef_x, 2, c2)));
        private_rsq = (((private_delx * private_delx) + (private_dely * private_dely)) + (private_delz * private_delz));
        if (private_rsq < cutforcesq) {
          private_sr2 = 1.0 / private_rsq;
          private_sr6 = ((private_sr2 * private_sr2) * private_sr2) * sigma6;
          private_force = (((48.0 * private_sr6) * (private_sr6 - 0.5)) * private_sr2) * epsilon;
          private_fix += (private_delx * private_force);
          private_fiy += (private_dely * private_force);
          private_fiz += (private_delz * private_force);
        }
      }
      for (int c2 = 32 * b0 + t0 + 1; c2 <= 8399; c2 += 1) {
        private_delx = (private_xtmp - (tex2D(texRef_x, 0, c2)));
        private_dely = (private_ytmp - (tex2D(texRef_x, 1, c2)));
        private_delz = (private_ztmp - (tex2D(texRef_x, 2, c2)));
        private_rsq = (((private_delx * private_delx) + (private_dely * private_dely)) + (private_delz * private_delz));
        if (private_rsq < cutforcesq) {
          private_sr2 = 1.0 / private_rsq;
          private_sr6 = ((private_sr2 * private_sr2) * private_sr2) * sigma6;
          private_force = (((48.0 * private_sr6) * (private_sr6 - 0.5)) * private_sr2) * epsilon;
          private_fix += (private_delx * private_force);
          private_fiy += (private_dely * private_force);
          private_fiz += (private_delz * private_force);
        }
      }
      f[32 * b0 + t0][1] += private_fiy;
      f[32 * b0 + t0][0] += private_fix;
      f[32 * b0 + t0][2] += private_fiz;
    }
}
