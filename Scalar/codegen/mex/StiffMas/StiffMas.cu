/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas.cu
 *
 * Code generation for function 'StiffMas'
 *
 */

/* Include files */
#include "StiffMas.h"
#include "MWCudaDimUtility.h"
#include "MWLaunchParametersUtilities.h"
#include "StiffMas_emxutil.h"
#include "introsort.h"
#include "rt_nonfinite.h"

/* Type Definitions */
struct StiffMas_kernel1_StackDataType
{
  uint32_T ind_1[64];
  real_T B_2[64];
};

/* Function Declarations */
static __global__ void StiffMas_kernel1(const real_T c, const real_T dv[8],
  const real_T dv1[8], const real_T dv2[8], const real_T nodes[3993], const
  uint32_T elements[8000], real_T Ke[64000], uint32_T jK[64000], uint32_T iK
  [64000], StiffMas_kernel1_StackDataType *globalStackData);
static __global__ void StiffMas_kernel10(const uint32_T b_data[128000], const
  int32_T b_size[2], uint32_T b_b_data[128000]);
static __global__ void StiffMas_kernel11(const int32_T nb, int32_T indx_size[1]);
static __global__ void StiffMas_kernel12(const int32_T idx[64000], const int32_T
  nb, int32_T sortedIndices_data[64000]);
static __global__ void StiffMas_kernel13(const uint16_T sz[2], int32_T r_data
  [64000]);
static __global__ void StiffMas_kernel14(const int32_T p, int32_T iwork[64000]);
static __global__ void StiffMas_kernel15(const int32_T sortedIndices_data[64000],
  const int32_T p, int32_T iwork[64000]);
static __global__ void StiffMas_kernel16(const int32_T indx_size[1], int32_T
  iwork[64000]);
static __global__ void StiffMas_kernel17(const int32_T iwork[64000], const
  int32_T idx_size[1], int32_T r_data[64000]);
static __global__ void StiffMas_kernel18(const int32_T idx[64000], const int32_T
  j, const int32_T kEnd, int32_T iwork[64000]);
static __global__ void StiffMas_kernel19(const uint32_T b_data[128000], const
  int32_T b_size[2], const int32_T r_data[64000], const int32_T b_b_size[2],
  const int32_T r_size[1], uint32_T b_b_data[128000]);
static __global__ void StiffMas_kernel2(const uint32_T jK[64000], const uint32_T
  iK[64000], uint32_T subs[128000]);
static __global__ void StiffMas_kernel20(const uint32_T b_data[128000], const
  int32_T b_size[2], uint32_T b_b_data[128000]);
static __global__ void StiffMas_kernel21(const int32_T r_data[64000], const
  int32_T nb, int32_T iwork[64000]);
static __global__ void StiffMas_kernel22(const int32_T iwork[64000], int32_T
  ipos[64000]);
static __global__ void StiffMas_kernel23(const int32_T ipos[64000], uint32_T
  ycol[64000]);
static __global__ void StiffMas_kernel24(const uint16_T sz[2], boolean_T
  filled_data[64000]);
static __global__ void StiffMas_kernel25(const uint16_T sz[2], real_T
  Afull_data[64000]);
static __global__ void StiffMas_kernel26(const int32_T b_size[2], int32_T
  ridxInt_size[1]);
static __global__ void StiffMas_kernel27(const uint32_T b_data[128000], const
  int32_T n, int32_T iwork[64000]);
static __global__ void StiffMas_kernel28(const int32_T b_size[2], int32_T
  cidxInt_size[1]);
static __global__ void StiffMas_kernel29(const uint32_T b_data[128000], const
  int32_T b_size[2], const int32_T n, int32_T idx[64000]);
static __global__ void StiffMas_kernel3(const uint32_T subs[128000], int32_T SZ
  [2]);
static __global__ void StiffMas_kernel30(const int32_T b_size[2], int32_T
  sortedIndices_size[1]);
static __global__ void StiffMas_kernel31(const int32_T pEnd, int32_T
  sortedIndices_data[64000]);
static __global__ void StiffMas_kernel32(const int32_T cidxInt_size[1],
  cell_wrap_1 tunableEnvironment[2]);
static __global__ void StiffMas_kernel33(const int32_T idx[64000], const int32_T
  cidxInt_size[1], cell_wrap_1 tunableEnvironment[2]);
static __global__ void StiffMas_kernel34(const int32_T ridxInt_size[1],
  cell_wrap_1 tunableEnvironment[2]);
static __global__ void StiffMas_kernel35(const int32_T iwork[64000], const
  int32_T ridxInt_size[1], cell_wrap_1 tunableEnvironment[2]);
static __global__ void StiffMas_kernel36(const cell_wrap_1 tunableEnvironment[2],
  const int32_T cidxInt_size[1], int32_T sortedIndices_data[64000]);
static __global__ void StiffMas_kernel37(const int32_T idx[64000], const int32_T
  cidxInt_size[1], int32_T r_data[64000]);
static __global__ void StiffMas_kernel38(const int32_T r_data[64000], const
  int32_T sortedIndices_data[64000], const int32_T n, int32_T idx[64000]);
static __global__ void StiffMas_kernel39(const int32_T iwork[64000], const
  int32_T ridxInt_size[1], int32_T r_data[64000]);
static __global__ void StiffMas_kernel4(const uint32_T subs[128000], const
  int32_T k, int32_T SZ[2]);
static __global__ void StiffMas_kernel40(const int32_T r_data[64000], const
  int32_T sortedIndices_data[64000], const int32_T n, int32_T iwork[64000]);
static __global__ void StiffMas_kernel41(const int32_T nb, const int32_T idx
  [64000], const int32_T n, const int32_T p, int32_T ipos[64000]);
static __global__ void StiffMas_kernel42(const int32_T n, const int32_T nb,
  int32_T idx[64000]);
static __global__ void StiffMas_kernel43(const int32_T iwork[64000], const
  int32_T j, const int32_T kEnd, int32_T idx[64000]);
static __global__ void StiffMas_kernel5(const uint32_T subs[128000], int32_T
  idx[64000]);
static __global__ void StiffMas_kernel6(const uint32_T subs[128000], const
  int32_T j, const int32_T idx[64000], uint32_T ycol[64000]);
static __global__ void StiffMas_kernel7(const uint32_T ycol[64000], const
  int32_T j, uint32_T subs[128000]);
static __global__ void StiffMas_kernel8(const uint32_T subs[128000], uint32_T
  b_data[128000]);
static __global__ void StiffMas_kernel9(const uint32_T b_data[128000], const
  int32_T b_size[2], const int32_T p, uint32_T b_b_data[128000]);

/* Function Definitions */
static __global__ __launch_bounds__(512, 1) void StiffMas_kernel1(const real_T c,
  const real_T dv[8], const real_T dv1[8], const real_T dv2[8], const real_T
  nodes[3993], const uint32_T elements[8000], real_T Ke[64000], uint32_T jK
  [64000], uint32_T iK[64000], StiffMas_kernel1_StackDataType *globalStackData)
{
  uint32_T threadId;
  int32_T k;
  int32_T i;
  int32_T r3;
  real_T X[24];
  int32_T e;
  int32_T jp1j;
  int32_T jA;
  int32_T b_i;
  real_T L[24];
  real_T d;
  real_T x[9];
  real_T Jac[9];
  int32_T j;
  int8_T ipiv[3];
  real_T detJ;
  int32_T b_c;
  int32_T c_c;
  boolean_T isodd;
  int32_T ix;
  real_T smax;
  real_T yjy;
  int32_T jy;
  int32_T initAuxVar;
  int32_T b_j;
  int32_T b_initAuxVar;
  real_T B[24];
  StiffMas_kernel1_StackDataType *mallocPtr;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  mallocPtr = globalStackData + threadId;
  e = static_cast<int32_T>(threadId);
  if (e < 1000) {
    /*  STIFFMAS Create the global stiffness matrix K for a SCALAR problem in SERIAL computing. */
    /*    STIFFMAS(elements,nodes,c) returns a sparse matrix K from finite element */
    /*    analysis of scalar problems in a three-dimensional domain, where "elements" */
    /*    is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of */
    /*    size Nx3, and "c" the material property for an isotropic material (scalar). */
    /*  */
    /*    See also STIFFMASS, STIFFMAPS, SPARSE */
    /*  */
    /*    For more information, see the <a href="matlab: */
    /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
    /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
    /*    Universidad Nacional de Colombia - Medellin */
    /*  	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved */
    /*  	Modified: 21/01/2019. Version: 1.3 */
    /*    Created:  30/11/2018. Version: 1.0 */
    /*  Add kernelfun pragma to trigger kernel creation */
    /* 'StiffMas:20' coder.gpu.kernelfun; */
    /* 'StiffMas:22' dTypeInd = class(elements); */
    /*  Data type (precision) for index computation */
    /* 'StiffMas:23' dTypeKe = class(nodes); */
    /*  Data type (precision) for ke computation */
    /* 'StiffMas:24' nel = size(elements,1); */
    /*  Total number of elements */
    /* 'StiffMas:25' iK = zeros(8,8,nel,dTypeInd); */
    /*  Stores the rows' indices */
    /* 'StiffMas:26' jK = zeros(8,8,nel,dTypeInd); */
    /*  Stores the columns' indices */
    /* 'StiffMas:27' Ke = zeros(8,8,nel,dTypeKe); */
    /*  Stores the NNZ values */
    /* 'StiffMas:28' for e = 1:nel */
    /*  Loop over elements */
    /* 'StiffMas:29' n = elements(e,:); */
    /*  Nodes of the element 'e' */
    /* 'StiffMas:30' X = nodes(n,:); */
    for (k = 0; k < 3; k++) {
      for (i = 0; i < 8; i++) {
        X[i + (k << 3)] = nodes[(static_cast<int32_T>(elements[e + 1000 * i]) +
          1331 * k) - 1];
      }
    }

    /*  Nodal coordinates of the element 'e' */
    /* 'StiffMas:31' ind = repmat(n,8,1); */
    for (r3 = 0; r3 < 8; r3++) {
      for (jp1j = 0; jp1j < 8; jp1j++) {
        mallocPtr->ind_1[(r3 << 3) + jp1j] = elements[e + 1000 * r3];
      }
    }

    /*  Index for element 'e' */
    /* 'StiffMas:32' iK(:,:,e) = ind'; */
    /*  Row index storage */
    /* 'StiffMas:33' jK(:,:,e) = ind; */
    /*  Columm index storage */
    /* 'StiffMas:34' Ke(:,:,e) = Hex8scalars(X,c); */
    /*  HEX8SCALARS Compute the element stiffnes matrix for a SCALAR problem in SERIAL computing */
    /*    HEX8SCALARS(X,c) returns the element stiffness matrix "ke" for an element */
    /*    "e"  in a finite element analysis of scalar problems in a three-dimensional */
    /*    domain computed in a serial manner on the CPU,  where "X" is the nodal */
    /*    coordinates of the element "e" (size 8x3), and "c" the material property */
    /*    (scalar). */
    /*  */
    /*    Examples: */
    /*          X = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1] */
    /*          ke = Hex8scalars(X,1) */
    /*   */
    /*    See also HEX8SCALARSAS, HEX8SCALARSAP */
    /*  */
    /*    For more information, see the <a href="matlab: */
    /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
    /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
    /*    Universidad Nacional de Colombia - Medellin */
    /*  	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved */
    /*  	Modified: 22/01/2019. Version: 1.3 */
    /*    Created:  30/11/2018. Version: 1.0 */
    /* 'Hex8scalars:24' p = 1/sqrt(3); */
    /*  Gauss points */
    /* 'Hex8scalars:25' r = [-p,p,p,-p,-p,p,p,-p]; */
    /*  Points through r-coordinate */
    /* 'Hex8scalars:26' s = [-p,-p,p,p,-p,-p,p,p]; */
    /*  Points through s-coordinate */
    /* 'Hex8scalars:27' t = [-p,-p,-p,-p,p,p,p,p]; */
    /*  Points through t-coordinate */
    /* 'Hex8scalars:28' ke = zeros(8,8); */
    for (k = 0; k < 8; k++) {
      for (i = 0; i < 8; i++) {
        iK[(i + (k << 3)) + (e << 6)] = mallocPtr->ind_1[k + (i << 3)];
        jK[(i + (k << 3)) + (e << 6)] = mallocPtr->ind_1[i + (k << 3)];
        Ke[(i + (k << 3)) + (e << 6)] = 0.0;
      }
    }

    /*  Initialize the element stiffness matrix */
    /* 'Hex8scalars:29' for i=1:8 */
    for (b_i = 0; b_i < 8; b_i++) {
      /*  Loop over numerical integration */
      /* 'Hex8scalars:30' ri = r(i); */
      /* 'Hex8scalars:30' si = s(i); */
      /* 'Hex8scalars:30' ti = t(i); */
      /*   Shape function derivatives with respect to r,s,t */
      /* 'Hex8scalars:32' dNdr = (1/8)*[-(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),... */
      /* 'Hex8scalars:33'         -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti)]; */
      /* 'Hex8scalars:34' dNds = (1/8)*[-(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),... */
      /* 'Hex8scalars:35'         -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti)]; */
      /* 'Hex8scalars:36' dNdt = (1/8)*[-(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),... */
      /* 'Hex8scalars:37'         (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)]; */
      /* 'Hex8scalars:38' L = [dNdr; dNds; dNdt]; */
      L[0] = 0.125 * (-(1.0 - dv2[b_i]) * (1.0 - dv1[b_i]));
      L[3] = 0.125 * ((1.0 - dv2[b_i]) * (1.0 - dv1[b_i]));
      L[6] = 0.125 * ((dv2[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[9] = 0.125 * (-(dv2[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[12] = 0.125 * (-(1.0 - dv2[b_i]) * (dv1[b_i] + 1.0));
      L[15] = 0.125 * ((1.0 - dv2[b_i]) * (dv1[b_i] + 1.0));
      L[18] = 0.125 * ((dv2[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[21] = 0.125 * (-(dv2[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[1] = 0.125 * (-(1.0 - dv[b_i]) * (1.0 - dv1[b_i]));
      L[4] = 0.125 * (-(dv[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[7] = 0.125 * ((dv[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[10] = 0.125 * ((1.0 - dv[b_i]) * (1.0 - dv1[b_i]));
      L[13] = 0.125 * (-(1.0 - dv[b_i]) * (dv1[b_i] + 1.0));
      L[16] = 0.125 * (-(dv[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[19] = 0.125 * ((dv[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[22] = 0.125 * ((1.0 - dv[b_i]) * (dv1[b_i] + 1.0));
      L[2] = 0.125 * (-(1.0 - dv[b_i]) * (1.0 - dv2[b_i]));
      L[5] = 0.125 * (-(dv[b_i] + 1.0) * (1.0 - dv2[b_i]));
      L[8] = 0.125 * (-(dv[b_i] + 1.0) * (dv2[b_i] + 1.0));
      L[11] = 0.125 * (-(1.0 - dv[b_i]) * (dv2[b_i] + 1.0));
      L[14] = 0.125 * ((1.0 - dv[b_i]) * (1.0 - dv2[b_i]));
      L[17] = 0.125 * ((dv[b_i] + 1.0) * (1.0 - dv2[b_i]));
      L[20] = 0.125 * ((dv[b_i] + 1.0) * (dv2[b_i] + 1.0));
      L[23] = 0.125 * ((1.0 - dv[b_i]) * (dv2[b_i] + 1.0));

      /*  L matrix */
      /* 'Hex8scalars:39' Jac  = L*X; */
      for (jA = 0; jA < 3; jA++) {
        for (k = 0; k < 3; k++) {
          d = 0.0;
          for (i = 0; i < 8; i++) {
            d += L[jA + 3 * i] * X[i + (k << 3)];
          }

          Jac[jA + 3 * k] = d;
        }
      }

      /*  Jacobian matrix */
      /* 'Hex8scalars:40' detJ = det(Jac); */
      for (i = 0; i < 9; i++) {
        x[i] = Jac[i];
      }

      for (i = 0; i < 3; i++) {
        ipiv[i] = static_cast<int8_T>((i + 1));
      }

      for (j = 0; j < 2; j++) {
        b_c = j << 2;
        jp1j = b_c + 2;
        c_c = 3 - j;
        r3 = 1;
        ix = b_c + 1;
        smax = fabs(x[b_c]);
        for (k = 0; k <= c_c - 2; k++) {
          ix++;
          yjy = fabs(x[ix - 1]);
          if (yjy > smax) {
            r3 = k + 2;
            smax = yjy;
          }
        }

        if (x[(b_c + r3) - 1] != 0.0) {
          if (r3 - 1 != 0) {
            ipiv[j] = static_cast<int8_T>((j + r3));
            initAuxVar = j + 1;
            b_initAuxVar = j + r3;
            for (k = 0; k < 3; k++) {
              ix = initAuxVar + k * 3;
              jA = b_initAuxVar + k * 3;
              yjy = x[ix - 1];
              x[ix - 1] = x[jA - 1];
              x[jA - 1] = yjy;
            }
          }

          i = (b_c - j) + 3;
          for (jA = 0; jA <= i - jp1j; jA++) {
            r3 = (b_c + jA) + 2;
            x[r3 - 1] /= x[b_c];
          }
        }

        c_c = 2 - j;
        jA = b_c + 4;
        jy = b_c + 4;
        for (b_j = 0; b_j < c_c; b_j++) {
          yjy = x[jy - 1];
          if (x[jy - 1] != 0.0) {
            ix = b_c + 2;
            k = jA + 1;
            i = (jA - j) + 2;
            for (jp1j = 0; jp1j <= i - k; jp1j++) {
              r3 = (jA + jp1j) + 1;
              x[r3 - 1] += x[ix - 1] * -yjy;
              ix++;
            }
          }

          jy += 3;
          jA += 3;
        }
      }

      detJ = 1.0;
      for (k = 0; k < 3; k++) {
        detJ *= x[k + 3 * k];
      }

      isodd = false;
      for (k = 0; k < 2; k++) {
        if (static_cast<int32_T>(ipiv[k]) > k + 1) {
          isodd = static_cast<boolean_T>(!static_cast<int32_T>(isodd));
        }
      }

      if (isodd) {
        detJ = -detJ;
      }

      /*  Jacobian's determinant */
      /* 'Hex8scalars:41' B = Jac\L; */
      jy = 1;
      jp1j = 2;
      r3 = 3;
      smax = fabs(Jac[0]);
      yjy = fabs(Jac[1]);
      if (yjy > smax) {
        smax = yjy;
        jy = 2;
        jp1j = 1;
      }

      if (fabs(Jac[2]) > smax) {
        jy = 3;
        jp1j = 2;
        r3 = 1;
      }

      Jac[jp1j - 1] /= Jac[jy - 1];
      Jac[r3 - 1] /= Jac[jy - 1];
      Jac[jp1j + 2] -= Jac[jp1j - 1] * Jac[jy + 2];
      Jac[r3 + 2] -= Jac[r3 - 1] * Jac[jy + 2];
      Jac[jp1j + 5] -= Jac[jp1j - 1] * Jac[jy + 5];
      Jac[r3 + 5] -= Jac[r3 - 1] * Jac[jy + 5];
      if (fabs(Jac[r3 + 2]) > fabs(Jac[jp1j + 2])) {
        jA = jp1j;
        jp1j = r3;
        r3 = jA;
      }

      Jac[r3 + 2] /= Jac[jp1j + 2];
      Jac[r3 + 5] -= Jac[r3 + 2] * Jac[jp1j + 5];
      for (k = 0; k < 8; k++) {
        d = L[(jy + 3 * k) - 1];
        yjy = L[(jp1j + 3 * k) - 1] - d * Jac[jp1j - 1];
        smax = ((L[(r3 + 3 * k) - 1] - d * Jac[r3 - 1]) - yjy * Jac[r3 + 2]) /
          Jac[r3 + 5];
        B[3 * k + 2] = smax;
        d -= smax * Jac[jy + 5];
        yjy -= smax * Jac[jp1j + 5];
        yjy /= Jac[jp1j + 2];
        B[3 * k + 1] = yjy;
        d -= yjy * Jac[jy + 2];
        d /= Jac[jy - 1];
        B[3 * k] = d;
      }

      /*  B matrix */
      /* 'Hex8scalars:42' ke = ke + c*detJ*(B'*B); */
      yjy = c * detJ;
      for (jA = 0; jA < 8; jA++) {
        for (k = 0; k < 8; k++) {
          d = 0.0;
          for (i = 0; i < 3; i++) {
            d += B[i + 3 * jA] * B[i + 3 * k];
          }

          mallocPtr->B_2[jA + (k << 3)] = d;
        }
      }

      for (k = 0; k < 8; k++) {
        for (i = 0; i < 8; i++) {
          Ke[(i + (k << 3)) + (e << 6)] += yjy * mallocPtr->B_2[i + (k << 3)];
        }
      }

      /*  Element stiffness matrix */
    }

    /*  Element stiffness matrix storage */
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel10(const
  uint32_T b_data[128000], const int32_T b_size[2], uint32_T b_b_data[128000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((b_size[0] * 2 - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    b_b_data[i] = b_data[i];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel11(const int32_T
  nb, int32_T indx_size[1])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    indx_size[0] = nb;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel12(const
  int32_T idx[64000], const int32_T nb, int32_T sortedIndices_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((nb - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    sortedIndices_data[k] = idx[k];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel13(const
  uint16_T sz[2], int32_T r_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((static_cast<int32_T>(sz[0]) - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    r_data[i] = 0;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel14(const
  int32_T p, int32_T iwork[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>(p);
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    iwork[i] = 0;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel15(const
  int32_T sortedIndices_data[64000], const int32_T p, int32_T iwork[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T jp1j;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>(((p - 1) / 2));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    jp1j = (k << 1) + 1;
    if (sortedIndices_data[jp1j - 1] <= sortedIndices_data[jp1j]) {
      iwork[jp1j - 1] = jp1j;
      iwork[jp1j] = jp1j + 1;
    } else {
      iwork[jp1j - 1] = jp1j + 1;
      iwork[jp1j] = jp1j;
    }
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel16(const int32_T
  indx_size[1], int32_T iwork[64000])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    iwork[indx_size[0] - 1] = indx_size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel17(const
  int32_T iwork[64000], const int32_T idx_size[1], int32_T r_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((idx_size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    r_data[i] = iwork[i];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel18(const
  int32_T idx[64000], const int32_T j, const int32_T kEnd, int32_T iwork[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((kEnd - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    iwork[(j + k) - 1] = idx[k];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel19(const
  uint32_T b_data[128000], const int32_T b_size[2], const int32_T r_data[64000],
  const int32_T b_b_size[2], const int32_T r_size[1], uint32_T b_b_data[128000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int32_T i;
  int64_T loopEnd;
  uint32_T tmpIndex;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = (static_cast<int64_T>((r_size[0] - 1)) + 1L) * 2L - 1L;
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>((idx % static_cast<uint32_T>(r_size[0])));
    tmpIndex = (idx - static_cast<uint32_T>(k)) / static_cast<uint32_T>(r_size[0]);
    i = static_cast<int32_T>(tmpIndex);
    b_b_data[k + b_b_size[0] * i] = b_data[(r_data[k] + b_size[0] * i) - 1];
  }
}

static __global__ __launch_bounds__(512, 1) void StiffMas_kernel2(const uint32_T
  jK[64000], const uint32_T iK[64000], uint32_T subs[128000])
{
  uint32_T threadId;
  int32_T i;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i = static_cast<int32_T>(threadId);
  if (i < 64000) {
    /* 'StiffMas:36' K = accumarray([iK(:),jK(:)],Ke(:),[],[],[],1); */
    subs[i] = iK[i];
    subs[i + 64000] = jK[i];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel20(const
  uint32_T b_data[128000], const int32_T b_size[2], uint32_T b_b_data[128000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((b_size[0] * 2 - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    b_b_data[i] = b_data[i];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel21(const
  int32_T r_data[64000], const int32_T nb, int32_T iwork[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((nb - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    iwork[r_data[k] - 1] = k + 1;
  }
}

static __global__ __launch_bounds__(512, 1) void StiffMas_kernel22(const int32_T
  iwork[64000], int32_T ipos[64000])
{
  uint32_T threadId;
  int32_T i;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i = static_cast<int32_T>(threadId);
  if (i < 64000) {
    ipos[i] = iwork[ipos[i] - 1];
  }
}

static __global__ __launch_bounds__(512, 1) void StiffMas_kernel23(const int32_T
  ipos[64000], uint32_T ycol[64000])
{
  uint32_T threadId;
  int32_T i;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i = static_cast<int32_T>(threadId);
  if (i < 64000) {
    ycol[i] = static_cast<uint32_T>(ipos[i]);
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel24(const
  uint16_T sz[2], boolean_T filled_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((static_cast<int32_T>(sz[0]) - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    filled_data[i] = true;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel25(const
  uint16_T sz[2], real_T Afull_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((static_cast<int32_T>(sz[0]) - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    Afull_data[i] = 0.0;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel26(const int32_T
  b_size[2], int32_T ridxInt_size[1])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    ridxInt_size[0] = b_size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel27(const
  uint32_T b_data[128000], const int32_T n, int32_T iwork[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((n - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    iwork[k] = static_cast<int32_T>(b_data[k]);
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel28(const int32_T
  b_size[2], int32_T cidxInt_size[1])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    cidxInt_size[0] = b_size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel29(const
  uint32_T b_data[128000], const int32_T b_size[2], const int32_T n, int32_T
  idx[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((n - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    idx[k] = static_cast<int32_T>(b_data[k + b_size[0]]);
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel3(const uint32_T
  subs[128000], int32_T SZ[2])
{
  uint32_T threadId;
  int32_T j;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  j = static_cast<int32_T>(threadId);
  if (j < 2) {
    SZ[j] = static_cast<int32_T>(subs[64000 * j]);
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel30(const int32_T
  b_size[2], int32_T sortedIndices_size[1])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    sortedIndices_size[0] = b_size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel31(const
  int32_T pEnd, int32_T sortedIndices_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((pEnd - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    sortedIndices_data[k] = k + 1;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel32(const int32_T
  cidxInt_size[1], cell_wrap_1 tunableEnvironment[2])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    tunableEnvironment[0].f1.size[0] = cidxInt_size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel33(const
  int32_T idx[64000], const int32_T cidxInt_size[1], cell_wrap_1
  tunableEnvironment[2])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((cidxInt_size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    i = static_cast<int32_T>(b_idx);
    tunableEnvironment[0].f1.data[i] = idx[i];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel34(const int32_T
  ridxInt_size[1], cell_wrap_1 tunableEnvironment[2])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    tunableEnvironment[1].f1.size[0] = ridxInt_size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel35(const
  int32_T iwork[64000], const int32_T ridxInt_size[1], cell_wrap_1
  tunableEnvironment[2])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((ridxInt_size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    tunableEnvironment[1].f1.data[i] = iwork[i];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel36(const
  cell_wrap_1 tunableEnvironment[2], const int32_T cidxInt_size[1], int32_T
  sortedIndices_data[64000])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    introsort(sortedIndices_data, cidxInt_size[0], tunableEnvironment);
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel37(const
  int32_T idx[64000], const int32_T cidxInt_size[1], int32_T r_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((cidxInt_size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    i = static_cast<int32_T>(b_idx);
    r_data[i] = idx[i];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel38(const
  int32_T r_data[64000], const int32_T sortedIndices_data[64000], const int32_T
  n, int32_T idx[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((n - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    idx[k] = r_data[sortedIndices_data[k] - 1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel39(const
  int32_T iwork[64000], const int32_T ridxInt_size[1], int32_T r_data[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((ridxInt_size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i = static_cast<int32_T>(idx);
    r_data[i] = iwork[i];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel4(const uint32_T
  subs[128000], const int32_T k, int32_T SZ[2])
{
  uint32_T threadId;
  uint32_T u;
  int32_T j;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  j = static_cast<int32_T>(threadId);
  if (j < 2) {
    u = subs[(k + 64000 * j) + 1];
    if (u > static_cast<uint32_T>(SZ[j])) {
      SZ[j] = static_cast<int32_T>(u);
    }
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel40(const
  int32_T r_data[64000], const int32_T sortedIndices_data[64000], const int32_T
  n, int32_T iwork[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((n - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    iwork[k] = r_data[sortedIndices_data[k] - 1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel41(const
  int32_T nb, const int32_T idx[64000], const int32_T n, const int32_T p,
  int32_T ipos[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T j;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((p - n));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    j = static_cast<int32_T>(b_idx);
    ipos[idx[(n + j) - 1] - 1] = nb;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel42(const int32_T
  n, const int32_T nb, int32_T idx[64000])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    idx[nb - 1] = idx[n - 1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel43(const
  int32_T iwork[64000], const int32_T j, const int32_T kEnd, int32_T idx[64000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((kEnd - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    idx[(j + k) - 1] = iwork[k];
  }
}

static __global__ __launch_bounds__(512, 1) void StiffMas_kernel5(const uint32_T
  subs[128000], int32_T idx[64000])
{
  uint32_T threadId;
  int32_T jA;
  int32_T k;
  boolean_T p;
  int32_T jp1j;
  boolean_T isodd;
  boolean_T exitg1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  k = static_cast<int32_T>(threadId);
  if (k < 32000) {
    jA = (k << 1) + 1;
    p = true;
    jp1j = 1;
    exitg1 = false;
    while ((!static_cast<int32_T>(exitg1)) && (static_cast<int32_T>((jp1j < 3))))
    {
      isodd = (subs[(jA + 64000 * (jp1j - 1)) - 1] == subs[jA + 64000 * (jp1j -
                1)]);
      if (!static_cast<int32_T>(isodd)) {
        p = (subs[(jA + 64000 * (jp1j - 1)) - 1] <= subs[jA + 64000 * (jp1j - 1)]);
        exitg1 = true;
      } else {
        jp1j++;
      }
    }

    if (p) {
      idx[jA - 1] = jA;
      idx[jA] = jA + 1;
    } else {
      idx[jA - 1] = jA + 1;
      idx[jA] = jA;
    }
  }
}

static __global__ __launch_bounds__(512, 1) void StiffMas_kernel6(const uint32_T
  subs[128000], const int32_T j, const int32_T idx[64000], uint32_T ycol[64000])
{
  uint32_T threadId;
  int32_T r3;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  r3 = static_cast<int32_T>(threadId);
  if (r3 < 64000) {
    ycol[r3] = subs[(idx[r3] + 64000 * j) - 1];
  }
}

static __global__ __launch_bounds__(512, 1) void StiffMas_kernel7(const uint32_T
  ycol[64000], const int32_T j, uint32_T subs[128000])
{
  uint32_T threadId;
  int32_T r3;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  r3 = static_cast<int32_T>(threadId);
  if (r3 < 64000) {
    subs[r3 + 64000 * j] = ycol[r3];
  }
}

static __global__ __launch_bounds__(512, 1) void StiffMas_kernel8(const uint32_T
  subs[128000], uint32_T b_data[128000])
{
  uint32_T threadId;
  int32_T i;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i = static_cast<int32_T>(threadId);
  if (i < 128000) {
    b_data[i] = subs[i];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMas_kernel9(const
  uint32_T b_data[128000], const int32_T b_size[2], const int32_T p, uint32_T
  b_b_data[128000])
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int32_T i;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = (static_cast<int64_T>(p) + 1L) * 2L - 1L;
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>((idx % (static_cast<uint32_T>(p) + 1U)));
    i = static_cast<int32_T>(((idx - static_cast<uint32_T>(k)) /
      (static_cast<uint32_T>(p) + 1U)));
    b_b_data[k + b_size[0] * i] = b_data[k + 64000 * i];
  }
}

/*
 * function K = StiffMas(elements,nodes,c)
 */
void StiffMas(StiffMasStackData *SD, const uint32_T elements[8000], const real_T
              nodes[3993], real_T c, coder_internal_sparse *K)
{
  int32_T SZ[2];
  int32_T k;
  int32_T i;
  int32_T j;
  int32_T i2;
  int32_T pEnd;
  int32_T b_size[2];
  int32_T p;
  int32_T q;
  int32_T qEnd;
  int32_T nb;
  int32_T kEnd;
  int32_T n;
  int32_T b_b_size[2];
  boolean_T b_p;
  int32_T b_c;
  int32_T indx_size[1];
  uint16_T sz[2];
  static const real_T dv[8] = { -0.57735026918962584, 0.57735026918962584,
    0.57735026918962584, -0.57735026918962584, -0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, -0.57735026918962584 };

  static const real_T dv1[8] = { -0.57735026918962584, -0.57735026918962584,
    -0.57735026918962584, -0.57735026918962584, 0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, 0.57735026918962584 };

  static const real_T dv2[8] = { -0.57735026918962584, -0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, -0.57735026918962584,
    -0.57735026918962584, 0.57735026918962584, 0.57735026918962584 };

  int32_T r_size[1];
  int32_T idx_size[1];
  int32_T c_b_size[2];
  uint16_T iwork_size[1];
  int32_T invr_size[1];
  boolean_T filled_data[64000];
  int32_T filled_size[2];
  uint16_T Afull_size[2];
  int32_T ridxInt_size[1];
  int32_T cidxInt_size[1];
  int32_T sortedIndices_size[1];
  real_T val;
  real_T (*gpu_dv)[8];
  real_T (*gpu_dv1)[8];
  real_T (*gpu_dv2)[8];
  real_T (*gpu_nodes)[3993];
  uint32_T (*gpu_elements)[8000];
  real_T (*gpu_Ke)[64000];
  uint32_T (*gpu_jK)[64000];
  uint32_T (*gpu_iK)[64000];
  uint32_T (*gpu_subs)[128000];
  int32_T (*gpu_SZ)[2];
  int32_T (*gpu_idx)[64000];
  uint32_T (*gpu_ycol)[64000];
  uint32_T (*gpu_b_data)[128000];
  int32_T (*gpu_b_size)[2];
  dim3 grid;
  dim3 block;
  boolean_T validLaunchParams;
  uint32_T (*b_gpu_b_data)[128000];
  dim3 b_grid;
  dim3 b_block;
  boolean_T b_validLaunchParams;
  int32_T (*gpu_indx_size)[1];
  dim3 c_grid;
  dim3 c_block;
  boolean_T c_validLaunchParams;
  int32_T (*gpu_sortedIndices_data)[64000];
  uint16_T (*gpu_sz)[2];
  dim3 d_grid;
  dim3 d_block;
  boolean_T d_validLaunchParams;
  int32_T (*gpu_r_data)[64000];
  dim3 e_grid;
  dim3 e_block;
  boolean_T e_validLaunchParams;
  int32_T (*gpu_iwork)[64000];
  dim3 f_grid;
  dim3 f_block;
  boolean_T f_validLaunchParams;
  int32_T (*gpu_idx_size)[1];
  dim3 g_grid;
  dim3 g_block;
  boolean_T g_validLaunchParams;
  dim3 h_grid;
  dim3 h_block;
  boolean_T h_validLaunchParams;
  int32_T (*b_gpu_b_size)[2];
  int32_T (*c_gpu_b_size)[2];
  int32_T (*gpu_r_size)[1];
  dim3 i_grid;
  dim3 i_block;
  boolean_T i_validLaunchParams;
  dim3 j_grid;
  dim3 j_block;
  boolean_T j_validLaunchParams;
  dim3 k_grid;
  dim3 k_block;
  boolean_T k_validLaunchParams;
  int32_T (*gpu_ipos)[64000];
  dim3 l_grid;
  dim3 l_block;
  boolean_T l_validLaunchParams;
  boolean_T (*gpu_filled_data)[64000];
  dim3 m_grid;
  dim3 m_block;
  boolean_T m_validLaunchParams;
  real_T (*gpu_Afull_data)[64000];
  int32_T (*gpu_ridxInt_size)[1];
  dim3 n_grid;
  dim3 n_block;
  boolean_T n_validLaunchParams;
  int32_T (*gpu_cidxInt_size)[1];
  dim3 o_grid;
  dim3 o_block;
  boolean_T o_validLaunchParams;
  int32_T (*gpu_sortedIndices_size)[1];
  dim3 p_grid;
  dim3 p_block;
  boolean_T p_validLaunchParams;
  cell_wrap_1 (*gpu_tunableEnvironment)[2];
  dim3 q_grid;
  dim3 q_block;
  boolean_T q_validLaunchParams;
  dim3 r_grid;
  dim3 r_block;
  boolean_T r_validLaunchParams;
  dim3 s_grid;
  dim3 s_block;
  boolean_T s_validLaunchParams;
  dim3 t_grid;
  dim3 t_block;
  boolean_T t_validLaunchParams;
  dim3 u_grid;
  dim3 u_block;
  boolean_T u_validLaunchParams;
  dim3 v_grid;
  dim3 v_block;
  boolean_T v_validLaunchParams;
  dim3 w_grid;
  dim3 w_block;
  boolean_T w_validLaunchParams;
  dim3 x_grid;
  dim3 x_block;
  boolean_T x_validLaunchParams;
  boolean_T Ke_dirtyOnGpu;
  boolean_T subs_dirtyOnGpu;
  boolean_T idx_dirtyOnGpu;
  boolean_T ycol_dirtyOnGpu;
  boolean_T b_data_dirtyOnGpu;
  boolean_T sortedIndices_data_dirtyOnGpu;
  boolean_T iwork_dirtyOnGpu;
  boolean_T filled_data_dirtyOnGpu;
  boolean_T Afull_data_dirtyOnGpu;
  boolean_T idx_dirtyOnCpu;
  boolean_T b_data_dirtyOnCpu;
  boolean_T b_size_dirtyOnCpu;
  boolean_T sz_dirtyOnCpu;
  boolean_T iwork_dirtyOnCpu;
  boolean_T b_b_size_dirtyOnCpu;
  StiffMas_kernel1_StackDataType *StiffMas_kernel1_StackData;
  boolean_T exitg1;
  int32_T exitg2;
  cudaMalloc(&gpu_tunableEnvironment, 512008UL);
  cudaMalloc(&gpu_sortedIndices_size, 4UL);
  cudaMalloc(&gpu_cidxInt_size, 4UL);
  cudaMalloc(&gpu_ridxInt_size, 4UL);
  cudaMalloc(&gpu_Afull_data, 64000U * sizeof(real_T));
  cudaMalloc(&gpu_Ke, 512000UL);
  cudaMalloc(&gpu_filled_data, 64000U * sizeof(boolean_T));
  cudaMalloc(&gpu_ipos, 256000UL);
  cudaMalloc(&c_gpu_b_size, 8UL);
  cudaMalloc(&gpu_r_data, 64000U * sizeof(int32_T));
  cudaMalloc(&gpu_sortedIndices_data, 64000U * sizeof(int32_T));
  cudaMalloc(&gpu_iwork, 64000U * sizeof(int32_T));
  cudaMalloc(&gpu_idx_size, 4UL);
  cudaMalloc(&gpu_sz, 4UL);
  cudaMalloc(&gpu_r_size, 4UL);
  cudaMalloc(&gpu_indx_size, 4UL);
  cudaMalloc(&b_gpu_b_data, 128000U * sizeof(uint32_T));
  cudaMalloc(&gpu_b_data, 128000U * sizeof(uint32_T));
  cudaMalloc(&gpu_b_size, 8UL);
  cudaMalloc(&b_gpu_b_size, 8UL);
  cudaMalloc(&gpu_ycol, 256000UL);
  cudaMalloc(&gpu_idx, 64000U * sizeof(int32_T));
  cudaMalloc(&gpu_SZ, 8UL);
  cudaMalloc(&gpu_subs, 512000UL);
  cudaMalloc(&gpu_iK, 256000UL);
  cudaMalloc(&gpu_jK, 256000UL);
  cudaMalloc(&gpu_elements, 32000UL);
  cudaMalloc(&gpu_nodes, 31944UL);
  cudaMalloc(&gpu_dv2, 64UL);
  cudaMalloc(&gpu_dv1, 64UL);
  cudaMalloc(&gpu_dv, 64UL);
  iwork_dirtyOnCpu = false;
  Afull_data_dirtyOnGpu = false;
  filled_data_dirtyOnGpu = false;
  iwork_dirtyOnGpu = false;
  sortedIndices_data_dirtyOnGpu = false;

  /*  STIFFMAS Create the global stiffness matrix K for a SCALAR problem in SERIAL computing. */
  /*    STIFFMAS(elements,nodes,c) returns a sparse matrix K from finite element */
  /*    analysis of scalar problems in a three-dimensional domain, where "elements" */
  /*    is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of */
  /*    size Nx3, and "c" the material property for an isotropic material (scalar). */
  /*  */
  /*    See also STIFFMASS, STIFFMAPS, SPARSE */
  /*  */
  /*    For more information, see the <a href="matlab: */
  /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
  /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
  /*    Universidad Nacional de Colombia - Medellin */
  /*  	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved */
  /*  	Modified: 21/01/2019. Version: 1.3 */
  /*    Created:  30/11/2018. Version: 1.0 */
  /*  Add kernelfun pragma to trigger kernel creation */
  /* 'StiffMas:20' coder.gpu.kernelfun; */
  /* 'StiffMas:22' dTypeInd = class(elements); */
  /*  Data type (precision) for index computation */
  /* 'StiffMas:23' dTypeKe = class(nodes); */
  /*  Data type (precision) for ke computation */
  /* 'StiffMas:24' nel = size(elements,1); */
  /*  Total number of elements */
  /* 'StiffMas:25' iK = zeros(8,8,nel,dTypeInd); */
  /*  Stores the rows' indices */
  /* 'StiffMas:26' jK = zeros(8,8,nel,dTypeInd); */
  /*  Stores the columns' indices */
  /* 'StiffMas:27' Ke = zeros(8,8,nel,dTypeKe); */
  /*  Stores the NNZ values */
  /* 'StiffMas:28' for e = 1:nel */
  cudaMemcpy(gpu_dv, (void *)&dv[0], 64UL, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_dv1, (void *)&dv1[0], 64UL, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_dv2, (void *)&dv2[0], 64UL, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nodes, (void *)&nodes[0], 31944UL, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_elements, (void *)&elements[0], 32000UL, cudaMemcpyHostToDevice);
  cudaMalloc(&StiffMas_kernel1_StackData, 1024U * sizeof
             (StiffMas_kernel1_StackDataType));
  StiffMas_kernel1<<<dim3(2U, 1U, 1U), dim3(512U, 1U, 1U)>>>(c, *gpu_dv,
    *gpu_dv1, *gpu_dv2, *gpu_nodes, *gpu_elements, *gpu_Ke, *gpu_jK, *gpu_iK,
    StiffMas_kernel1_StackData);
  cudaFree(StiffMas_kernel1_StackData);
  Ke_dirtyOnGpu = true;

  /* 'StiffMas:36' K = accumarray([iK(:),jK(:)],Ke(:),[],[],[],1); */
  StiffMas_kernel2<<<dim3(125U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_jK, *gpu_iK,
    *gpu_subs);
  subs_dirtyOnGpu = true;
  StiffMas_kernel3<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_subs, *gpu_SZ);
  for (k = 0; k < 63999; k++) {
    StiffMas_kernel4<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_subs, k,
      *gpu_SZ);
  }

  StiffMas_kernel5<<<dim3(63U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_subs,
    *gpu_idx);
  idx_dirtyOnCpu = false;
  idx_dirtyOnGpu = true;
  i = 2;
  while (i < 64000) {
    i2 = i << 1;
    j = 1;
    for (pEnd = i + 1; pEnd < 64001; pEnd = qEnd + i) {
      p = j;
      q = pEnd;
      qEnd = j + i2;
      if (qEnd > 64001) {
        qEnd = 64001;
      }

      k = 0;
      kEnd = qEnd - j;
      while (k + 1 <= kEnd) {
        if (idx_dirtyOnGpu) {
          cudaMemcpy(&SD->f0.idx[0], gpu_idx, 256000UL, cudaMemcpyDeviceToHost);
          idx_dirtyOnGpu = false;
        }

        n = SD->f0.idx[p - 1] - 1;
        nb = SD->f0.idx[q - 1] - 1;
        b_p = true;
        b_c = 0;
        exitg1 = false;
        while ((!exitg1) && (b_c + 1 < 3)) {
          if (subs_dirtyOnGpu) {
            cudaMemcpy(&SD->f0.subs[0], gpu_subs, 512000UL,
                       cudaMemcpyDeviceToHost);
            subs_dirtyOnGpu = false;
          }

          if (SD->f0.subs[n + 64000 * b_c] != SD->f0.subs[nb + 64000 * b_c]) {
            b_p = (SD->f0.subs[n + 64000 * b_c] <= SD->f0.subs[nb + 64000 * b_c]);
            exitg1 = true;
          } else {
            b_c++;
          }
        }

        if (b_p) {
          SD->f0.iwork[k] = SD->f0.idx[p - 1];
          iwork_dirtyOnCpu = true;
          p++;
          if (p == pEnd) {
            while (q < qEnd) {
              k++;
              SD->f0.iwork[k] = SD->f0.idx[q - 1];
              q++;
            }
          }
        } else {
          SD->f0.iwork[k] = SD->f0.idx[q - 1];
          iwork_dirtyOnCpu = true;
          q++;
          if (q == qEnd) {
            while (p < pEnd) {
              k++;
              SD->f0.iwork[k] = SD->f0.idx[p - 1];
              p++;
            }
          }
        }

        k++;
      }

      x_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((kEnd - 1)
        + 1L)), &x_grid, &x_block, 1024U, 65535U);
      if (x_validLaunchParams) {
        if (iwork_dirtyOnCpu) {
          cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], 256000UL,
                     cudaMemcpyHostToDevice);
          iwork_dirtyOnCpu = false;
        }

        StiffMas_kernel43<<<x_grid, x_block>>>(*gpu_iwork, j, kEnd, *gpu_idx);
        idx_dirtyOnGpu = true;
      }

      j = qEnd;
    }

    i = i2;
  }

  for (j = 0; j < 2; j++) {
    StiffMas_kernel6<<<dim3(125U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_subs, j,
      *gpu_idx, *gpu_ycol);
    StiffMas_kernel7<<<dim3(125U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_ycol, j,
      *gpu_subs);
  }

  b_size[0] = 64000;
  b_size[1] = 2;
  StiffMas_kernel8<<<dim3(250U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_subs,
    *gpu_b_data);
  b_data_dirtyOnCpu = false;
  b_data_dirtyOnGpu = true;
  nb = 0;
  k = 1;
  while (k <= 64000) {
    n = k;
    do {
      exitg2 = 0;
      k++;
      if (k > 64000) {
        exitg2 = 1;
      } else {
        b_p = false;
        j = 0;
        exitg1 = false;
        while ((!exitg1) && (j < 2)) {
          if (b_data_dirtyOnGpu) {
            cudaMemcpy(&SD->f0.b_data[0], gpu_b_data, b_size[0] * b_size[1] *
                       sizeof(uint32_T), cudaMemcpyDeviceToHost);
            b_data_dirtyOnGpu = false;
          }

          if (SD->f0.b_data[(n + 64000 * j) - 1] != SD->f0.b_data[(k + 64000 * j)
              - 1]) {
            b_p = true;
            exitg1 = true;
          } else {
            j++;
          }
        }

        if (b_p) {
          exitg2 = 1;
        }
      }
    } while (exitg2 == 0);

    nb++;
    for (j = 0; j < 2; j++) {
      if (b_data_dirtyOnGpu) {
        cudaMemcpy(&SD->f0.b_data[0], gpu_b_data, b_size[0] * b_size[1] * sizeof
                   (uint32_T), cudaMemcpyDeviceToHost);
        b_data_dirtyOnGpu = false;
      }

      SD->f0.b_data[(nb + 64000 * j) - 1] = SD->f0.b_data[(n + 64000 * j) - 1];
      b_data_dirtyOnCpu = true;
    }

    p = k - 1;
    w_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((p - n) +
      1L)), &w_grid, &w_block, 1024U, 65535U);
    if (w_validLaunchParams) {
      StiffMas_kernel41<<<w_grid, w_block>>>(nb, *gpu_idx, n, p, *gpu_ipos);
    }

    StiffMas_kernel42<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(n, nb, *gpu_idx);
    idx_dirtyOnGpu = true;
  }

  if (1 > nb) {
    p = -1;
  } else {
    p = nb - 1;
  }

  b_b_size[0] = p + 1;
  b_b_size[1] = 2;
  b_size_dirtyOnCpu = true;
  validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((p + 1L) * 2L)),
    &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    if (b_data_dirtyOnCpu) {
      cudaMemcpy(gpu_b_data, &SD->f0.b_data[0], b_size[0] * b_size[1] * sizeof
                 (uint32_T), cudaMemcpyHostToDevice);
      b_data_dirtyOnCpu = false;
    }

    cudaMemcpy(gpu_b_size, &b_b_size[0], 8UL, cudaMemcpyHostToDevice);
    b_size_dirtyOnCpu = false;
    StiffMas_kernel9<<<grid, block>>>(*gpu_b_data, *gpu_b_size, p, *b_gpu_b_data);
  }

  b_size[0] = b_b_size[0];
  b_size[1] = 2;
  b_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((b_b_size[0] *
    2 - 1) + 1L)), &b_grid, &b_block, 1024U, 65535U);
  if (b_validLaunchParams) {
    if (b_data_dirtyOnCpu) {
      cudaMemcpy(gpu_b_data, &SD->f0.b_data[0], b_size[0] * b_size[1] * sizeof
                 (uint32_T), cudaMemcpyHostToDevice);
      b_data_dirtyOnCpu = false;
    }

    if (b_size_dirtyOnCpu) {
      cudaMemcpy(gpu_b_size, &b_b_size[0], 8UL, cudaMemcpyHostToDevice);
    }

    StiffMas_kernel10<<<b_grid, b_block>>>(*b_gpu_b_data, *gpu_b_size,
      *gpu_b_data);
  }

  StiffMas_kernel11<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(nb, *gpu_indx_size);
  cudaMemcpy(&indx_size[0], gpu_indx_size, 4UL, cudaMemcpyDeviceToHost);
  c_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((nb - 1) + 1L)),
    &c_grid, &c_block, 1024U, 65535U);
  if (c_validLaunchParams) {
    StiffMas_kernel12<<<c_grid, c_block>>>(*gpu_idx, nb, *gpu_sortedIndices_data);
    sortedIndices_data_dirtyOnGpu = true;
  }

  n = indx_size[0] + 1;
  sz[0] = static_cast<uint16_T>(indx_size[0]);
  r_size[0] = static_cast<uint16_T>(indx_size[0]);
  d_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((sz[0] - 1) +
    1L)), &d_grid, &d_block, 1024U, 65535U);
  if (d_validLaunchParams) {
    cudaMemcpy(gpu_sz, &sz[0], 4UL, cudaMemcpyHostToDevice);
    StiffMas_kernel13<<<d_grid, d_block>>>(*gpu_sz, *gpu_r_data);
  }

  if (indx_size[0] != 0) {
    p = sz[0] - 1;
    idx_size[0] = sz[0];
    e_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>((p + 1L)),
      &e_grid, &e_block, 1024U, 65535U);
    if (e_validLaunchParams) {
      if (iwork_dirtyOnCpu) {
        cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], idx_size[0] * sizeof(int32_T),
                   cudaMemcpyHostToDevice);
        iwork_dirtyOnCpu = false;
      }

      StiffMas_kernel14<<<e_grid, e_block>>>(p, *gpu_iwork);
      iwork_dirtyOnGpu = true;
    }

    iwork_size[0] = sz[0];
    p = indx_size[0] - 1;
    f_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((p - 1) / 2
      + 1L)), &f_grid, &f_block, 1024U, 65535U);
    if (f_validLaunchParams) {
      if (iwork_dirtyOnCpu) {
        cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], idx_size[0] * sizeof(int32_T),
                   cudaMemcpyHostToDevice);
        iwork_dirtyOnCpu = false;
      }

      StiffMas_kernel15<<<f_grid, f_block>>>(*gpu_sortedIndices_data, p,
        *gpu_iwork);
      iwork_dirtyOnGpu = true;
    }

    if ((indx_size[0] & 1) != 0) {
      if (iwork_dirtyOnCpu) {
        cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], idx_size[0] * sizeof(int32_T),
                   cudaMemcpyHostToDevice);
        iwork_dirtyOnCpu = false;
      }

      StiffMas_kernel16<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_indx_size,
        *gpu_iwork);
      iwork_dirtyOnGpu = true;
    }

    i = 2;
    while (i < n - 1) {
      i2 = i << 1;
      j = 1;
      for (pEnd = i + 1; pEnd < n; pEnd = qEnd + i) {
        p = j;
        q = pEnd;
        qEnd = j + i2;
        if (qEnd > n) {
          qEnd = n;
        }

        k = 0;
        kEnd = qEnd - j;
        while (k + 1 <= kEnd) {
          if (sortedIndices_data_dirtyOnGpu) {
            cudaMemcpy(&SD->f0.sortedIndices_data[0], gpu_sortedIndices_data,
                       indx_size[0] * sizeof(int32_T), cudaMemcpyDeviceToHost);
            sortedIndices_data_dirtyOnGpu = false;
          }

          if (iwork_dirtyOnGpu) {
            cudaMemcpy(&SD->f0.iwork[0], gpu_iwork, idx_size[0] * sizeof(int32_T),
                       cudaMemcpyDeviceToHost);
            iwork_dirtyOnGpu = false;
          }

          if (SD->f0.sortedIndices_data[SD->f0.iwork[p - 1] - 1] <=
              SD->f0.sortedIndices_data[SD->f0.iwork[q - 1] - 1]) {
            if (idx_dirtyOnGpu) {
              cudaMemcpy(&SD->f0.idx[0], gpu_idx, iwork_size[0] * sizeof(int32_T),
                         cudaMemcpyDeviceToHost);
              idx_dirtyOnGpu = false;
            }

            SD->f0.idx[k] = SD->f0.iwork[p - 1];
            idx_dirtyOnCpu = true;
            p++;
            if (p == pEnd) {
              while (q < qEnd) {
                k++;
                SD->f0.idx[k] = SD->f0.iwork[q - 1];
                q++;
              }
            }
          } else {
            if (idx_dirtyOnGpu) {
              cudaMemcpy(&SD->f0.idx[0], gpu_idx, iwork_size[0] * sizeof(int32_T),
                         cudaMemcpyDeviceToHost);
              idx_dirtyOnGpu = false;
            }

            SD->f0.idx[k] = SD->f0.iwork[q - 1];
            idx_dirtyOnCpu = true;
            q++;
            if (q == qEnd) {
              while (p < pEnd) {
                k++;
                SD->f0.idx[k] = SD->f0.iwork[p - 1];
                p++;
              }
            }
          }

          k++;
        }

        h_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((kEnd -
          1) + 1L)), &h_grid, &h_block, 1024U, 65535U);
        if (h_validLaunchParams) {
          if (idx_dirtyOnCpu) {
            cudaMemcpy(gpu_idx, &SD->f0.idx[0], iwork_size[0] * sizeof(int32_T),
                       cudaMemcpyHostToDevice);
            idx_dirtyOnCpu = false;
          }

          if (iwork_dirtyOnCpu) {
            cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], idx_size[0] * sizeof(int32_T),
                       cudaMemcpyHostToDevice);
            iwork_dirtyOnCpu = false;
          }

          StiffMas_kernel18<<<h_grid, h_block>>>(*gpu_idx, j, kEnd, *gpu_iwork);
          iwork_dirtyOnGpu = true;
        }

        j = qEnd;
      }

      i = i2;
    }

    g_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((idx_size[0]
      - 1) + 1L)), &g_grid, &g_block, 1024U, 65535U);
    if (g_validLaunchParams) {
      if (iwork_dirtyOnCpu) {
        cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], idx_size[0] * sizeof(int32_T),
                   cudaMemcpyHostToDevice);
        iwork_dirtyOnCpu = false;
      }

      cudaMemcpy(gpu_idx_size, &idx_size[0], 4UL, cudaMemcpyHostToDevice);
      StiffMas_kernel17<<<g_grid, g_block>>>(*gpu_iwork, *gpu_idx_size,
        *gpu_r_data);
    }
  }

  c_b_size[0] = r_size[0];
  c_b_size[1] = 2;
  b_b_size_dirtyOnCpu = true;
  i_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>((((r_size[0] -
    1) + 1L) * 2L)), &i_grid, &i_block, 1024U, 65535U);
  if (i_validLaunchParams) {
    if (b_data_dirtyOnCpu) {
      cudaMemcpy(gpu_b_data, &SD->f0.b_data[0], b_size[0] * b_size[1] * sizeof
                 (uint32_T), cudaMemcpyHostToDevice);
      b_data_dirtyOnCpu = false;
    }

    cudaMemcpy(b_gpu_b_size, &b_size[0], 8UL, cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu_b_size, &c_b_size[0], 8UL, cudaMemcpyHostToDevice);
    b_b_size_dirtyOnCpu = false;
    cudaMemcpy(gpu_r_size, &r_size[0], 4UL, cudaMemcpyHostToDevice);
    StiffMas_kernel19<<<i_grid, i_block>>>(*gpu_b_data, *b_gpu_b_size,
      *gpu_r_data, *c_gpu_b_size, *gpu_r_size, *b_gpu_b_data);
  }

  b_size[0] = c_b_size[0];
  b_size[1] = 2;
  j_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((c_b_size[0] *
    2 - 1) + 1L)), &j_grid, &j_block, 1024U, 65535U);
  if (j_validLaunchParams) {
    if (b_data_dirtyOnCpu) {
      cudaMemcpy(gpu_b_data, &SD->f0.b_data[0], b_size[0] * b_size[1] * sizeof
                 (uint32_T), cudaMemcpyHostToDevice);
      b_data_dirtyOnCpu = false;
    }

    if (b_b_size_dirtyOnCpu) {
      cudaMemcpy(c_gpu_b_size, &c_b_size[0], 8UL, cudaMemcpyHostToDevice);
    }

    StiffMas_kernel20<<<j_grid, j_block>>>(*b_gpu_b_data, *c_gpu_b_size,
      *gpu_b_data);
  }

  invr_size[0] = r_size[0];
  k_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((nb - 1) + 1L)),
    &k_grid, &k_block, 1024U, 65535U);
  if (k_validLaunchParams) {
    if (iwork_dirtyOnCpu) {
      cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], invr_size[0] * sizeof(int32_T),
                 cudaMemcpyHostToDevice);
      iwork_dirtyOnCpu = false;
    }

    StiffMas_kernel21<<<k_grid, k_block>>>(*gpu_r_data, nb, *gpu_iwork);
    iwork_dirtyOnGpu = true;
  }

  if (iwork_dirtyOnCpu) {
    cudaMemcpy(gpu_iwork, &SD->f0.iwork[0], invr_size[0] * sizeof(int32_T),
               cudaMemcpyHostToDevice);
  }

  StiffMas_kernel22<<<dim3(125U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_iwork,
    *gpu_ipos);
  StiffMas_kernel23<<<dim3(125U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_ipos,
    *gpu_ycol);
  ycol_dirtyOnGpu = true;
  sz[0] = static_cast<uint16_T>(b_size[0]);
  sz_dirtyOnCpu = true;
  filled_size[0] = b_size[0];
  l_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((sz[0] - 1) +
    1L)), &l_grid, &l_block, 1024U, 65535U);
  if (l_validLaunchParams) {
    cudaMemcpy(gpu_sz, &sz[0], 4UL, cudaMemcpyHostToDevice);
    sz_dirtyOnCpu = false;
    StiffMas_kernel24<<<l_grid, l_block>>>(*gpu_sz, *gpu_filled_data);
    filled_data_dirtyOnGpu = true;
  }

  Afull_size[0] = sz[0];
  m_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((sz[0] - 1) +
    1L)), &m_grid, &m_block, 1024U, 65535U);
  if (m_validLaunchParams) {
    if (sz_dirtyOnCpu) {
      cudaMemcpy(gpu_sz, &sz[0], 4UL, cudaMemcpyHostToDevice);
    }

    StiffMas_kernel25<<<m_grid, m_block>>>(*gpu_sz, *gpu_Afull_data);
    Afull_data_dirtyOnGpu = true;
  }

  for (k = 0; k < 64000; k++) {
    if (ycol_dirtyOnGpu) {
      cudaMemcpy(&SD->f0.ycol[0], gpu_ycol, 256000UL, cudaMemcpyDeviceToHost);
      ycol_dirtyOnGpu = false;
    }

    if (filled_data_dirtyOnGpu) {
      cudaMemcpy(&filled_data[0], gpu_filled_data, filled_size[0] * sizeof
                 (boolean_T), cudaMemcpyDeviceToHost);
      filled_data_dirtyOnGpu = false;
    }

    if (filled_data[static_cast<int32_T>(SD->f0.ycol[k]) - 1]) {
      filled_data[static_cast<int32_T>(SD->f0.ycol[k]) - 1] = false;
      if (Ke_dirtyOnGpu) {
        cudaMemcpy(&SD->f0.Ke[0], gpu_Ke, 512000UL, cudaMemcpyDeviceToHost);
        Ke_dirtyOnGpu = false;
      }

      if (Afull_data_dirtyOnGpu) {
        cudaMemcpy(&SD->f0.Afull_data[0], gpu_Afull_data, Afull_size[0] * sizeof
                   (real_T), cudaMemcpyDeviceToHost);
        Afull_data_dirtyOnGpu = false;
      }

      SD->f0.Afull_data[static_cast<int32_T>(SD->f0.ycol[k]) - 1] = SD->f0.Ke[k];
    } else {
      if (Ke_dirtyOnGpu) {
        cudaMemcpy(&SD->f0.Ke[0], gpu_Ke, 512000UL, cudaMemcpyDeviceToHost);
        Ke_dirtyOnGpu = false;
      }

      if (Afull_data_dirtyOnGpu) {
        cudaMemcpy(&SD->f0.Afull_data[0], gpu_Afull_data, Afull_size[0] * sizeof
                   (real_T), cudaMemcpyDeviceToHost);
        Afull_data_dirtyOnGpu = false;
      }

      SD->f0.Afull_data[static_cast<int32_T>(SD->f0.ycol[k]) - 1] += SD->f0.Ke[k];
    }
  }

  pEnd = b_size[0];
  n = b_size[0];
  cudaMemcpy(b_gpu_b_size, &b_size[0], 8UL, cudaMemcpyHostToDevice);
  StiffMas_kernel26<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*b_gpu_b_size,
    *gpu_ridxInt_size);
  cudaMemcpy(&ridxInt_size[0], gpu_ridxInt_size, 4UL, cudaMemcpyDeviceToHost);
  n_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((n - 1) + 1L)),
    &n_grid, &n_block, 1024U, 65535U);
  if (n_validLaunchParams) {
    if (b_data_dirtyOnCpu) {
      cudaMemcpy(gpu_b_data, &SD->f0.b_data[0], b_size[0] * b_size[1] * sizeof
                 (uint32_T), cudaMemcpyHostToDevice);
      b_data_dirtyOnCpu = false;
    }

    StiffMas_kernel27<<<n_grid, n_block>>>(*gpu_b_data, n, *gpu_iwork);
    iwork_dirtyOnGpu = true;
  }

  n = b_size[0];
  StiffMas_kernel28<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*b_gpu_b_size,
    *gpu_cidxInt_size);
  cudaMemcpy(&cidxInt_size[0], gpu_cidxInt_size, 4UL, cudaMemcpyDeviceToHost);
  o_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((n - 1) + 1L)),
    &o_grid, &o_block, 1024U, 65535U);
  if (o_validLaunchParams) {
    if (idx_dirtyOnCpu) {
      cudaMemcpy(gpu_idx, &SD->f0.idx[0], cidxInt_size[0] * sizeof(int32_T),
                 cudaMemcpyHostToDevice);
      idx_dirtyOnCpu = false;
    }

    if (b_data_dirtyOnCpu) {
      cudaMemcpy(gpu_b_data, &SD->f0.b_data[0], b_size[0] * b_size[1] * sizeof
                 (uint32_T), cudaMemcpyHostToDevice);
    }

    StiffMas_kernel29<<<o_grid, o_block>>>(*gpu_b_data, *b_gpu_b_size, n,
      *gpu_idx);
    idx_dirtyOnGpu = true;
  }

  StiffMas_kernel30<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*b_gpu_b_size,
    *gpu_sortedIndices_size);
  cudaMemcpy(&sortedIndices_size[0], gpu_sortedIndices_size, 4UL,
             cudaMemcpyDeviceToHost);
  p_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((pEnd - 1) +
    1L)), &p_grid, &p_block, 1024U, 65535U);
  if (p_validLaunchParams) {
    StiffMas_kernel31<<<p_grid, p_block>>>(pEnd, *gpu_sortedIndices_data);
  }

  StiffMas_kernel32<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_cidxInt_size, *
    gpu_tunableEnvironment);
  q_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
    (((cidxInt_size[0] - 1) + 1L)), &q_grid, &q_block, 1024U, 65535U);
  if (q_validLaunchParams) {
    if (idx_dirtyOnCpu) {
      cudaMemcpy(gpu_idx, &SD->f0.idx[0], cidxInt_size[0] * sizeof(int32_T),
                 cudaMemcpyHostToDevice);
      idx_dirtyOnCpu = false;
    }

    StiffMas_kernel33<<<q_grid, q_block>>>(*gpu_idx, *gpu_cidxInt_size,
      *gpu_tunableEnvironment);
  }

  StiffMas_kernel34<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_ridxInt_size, *
    gpu_tunableEnvironment);
  r_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
    (((ridxInt_size[0] - 1) + 1L)), &r_grid, &r_block, 1024U, 65535U);
  if (r_validLaunchParams) {
    StiffMas_kernel35<<<r_grid, r_block>>>(*gpu_iwork, *gpu_ridxInt_size,
      *gpu_tunableEnvironment);
  }

  StiffMas_kernel36<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
    (*gpu_tunableEnvironment, *gpu_cidxInt_size, *gpu_sortedIndices_data);
  sortedIndices_data_dirtyOnGpu = true;
  n = cidxInt_size[0];
  s_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
    (((cidxInt_size[0] - 1) + 1L)), &s_grid, &s_block, 1024U, 65535U);
  if (s_validLaunchParams) {
    if (idx_dirtyOnCpu) {
      cudaMemcpy(gpu_idx, &SD->f0.idx[0], cidxInt_size[0] * sizeof(int32_T),
                 cudaMemcpyHostToDevice);
      idx_dirtyOnCpu = false;
    }

    StiffMas_kernel37<<<s_grid, s_block>>>(*gpu_idx, *gpu_cidxInt_size,
      *gpu_r_data);
  }

  cudaMemcpy(&sortedIndices_size[0], gpu_sortedIndices_size, 4UL,
             cudaMemcpyDeviceToHost);
  t_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((n - 1) + 1L)),
    &t_grid, &t_block, 1024U, 65535U);
  if (t_validLaunchParams) {
    if (idx_dirtyOnCpu) {
      cudaMemcpy(gpu_idx, &SD->f0.idx[0], cidxInt_size[0] * sizeof(int32_T),
                 cudaMemcpyHostToDevice);
    }

    StiffMas_kernel38<<<t_grid, t_block>>>(*gpu_r_data, *gpu_sortedIndices_data,
      n, *gpu_idx);
    idx_dirtyOnGpu = true;
  }

  n = ridxInt_size[0];
  u_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
    (((ridxInt_size[0] - 1) + 1L)), &u_grid, &u_block, 1024U, 65535U);
  if (u_validLaunchParams) {
    StiffMas_kernel39<<<u_grid, u_block>>>(*gpu_iwork, *gpu_ridxInt_size,
      *gpu_r_data);
  }

  v_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((n - 1) + 1L)),
    &v_grid, &v_block, 1024U, 65535U);
  if (v_validLaunchParams) {
    StiffMas_kernel40<<<v_grid, v_block>>>(*gpu_r_data, *gpu_sortedIndices_data,
      n, *gpu_iwork);
    iwork_dirtyOnGpu = true;
  }

  cudaMemcpy(&SZ[0], gpu_SZ, 8UL, cudaMemcpyDeviceToHost);
  K->m = SZ[0];
  K->n = SZ[1];
  if (b_size[0] >= 1) {
    n = b_size[0];
  } else {
    n = 1;
  }

  p = K->d->size[0];
  K->d->size[0] = n;
  emxEnsureCapacity_real_T(K->d, p);
  for (p = 0; p < n; p++) {
    K->d->data[p] = 0.0;
  }

  K->maxnz = n;
  p = K->colidx->size[0];
  K->colidx->size[0] = SZ[1] + 1;
  emxEnsureCapacity_int32_T(K->colidx, p);
  K->colidx->data[0] = 1;
  p = K->rowidx->size[0];
  K->rowidx->size[0] = n;
  emxEnsureCapacity_int32_T(K->rowidx, p);
  for (p = 0; p < n; p++) {
    K->rowidx->data[p] = 0;
  }

  n = 0;
  p = SZ[1];
  for (b_c = 0; b_c < p; b_c++) {
    exitg1 = false;
    while ((!exitg1) && (n + 1 <= b_size[0])) {
      if (idx_dirtyOnGpu) {
        cudaMemcpy(&SD->f0.idx[0], gpu_idx, cidxInt_size[0] * sizeof(int32_T),
                   cudaMemcpyDeviceToHost);
        idx_dirtyOnGpu = false;
      }

      if (SD->f0.idx[n] == b_c + 1) {
        if (iwork_dirtyOnGpu) {
          cudaMemcpy(&SD->f0.iwork[0], gpu_iwork, ridxInt_size[0] * sizeof
                     (int32_T), cudaMemcpyDeviceToHost);
          iwork_dirtyOnGpu = false;
        }

        K->rowidx->data[n] = SD->f0.iwork[n];
        n++;
      } else {
        exitg1 = true;
      }
    }

    K->colidx->data[b_c + 1] = n + 1;
  }

  for (k = 0; k < pEnd; k++) {
    if (sortedIndices_data_dirtyOnGpu) {
      cudaMemcpy(&SD->f0.sortedIndices_data[0], gpu_sortedIndices_data,
                 sortedIndices_size[0] * sizeof(int32_T), cudaMemcpyDeviceToHost);
      sortedIndices_data_dirtyOnGpu = false;
    }

    if (Afull_data_dirtyOnGpu) {
      cudaMemcpy(&SD->f0.Afull_data[0], gpu_Afull_data, Afull_size[0] * sizeof
                 (real_T), cudaMemcpyDeviceToHost);
      Afull_data_dirtyOnGpu = false;
    }

    K->d->data[k] = SD->f0.Afull_data[SD->f0.sortedIndices_data[k] - 1];
  }

  n = 1;
  p = K->colidx->size[0];
  for (b_c = 0; b_c <= p - 2; b_c++) {
    pEnd = K->colidx->data[b_c];
    K->colidx->data[b_c] = n;
    while (pEnd < K->colidx->data[b_c + 1]) {
      val = 0.0;
      nb = K->rowidx->data[pEnd - 1];
      while ((pEnd < K->colidx->data[b_c + 1]) && (K->rowidx->data[pEnd - 1] ==
              nb)) {
        val += K->d->data[pEnd - 1];
        pEnd++;
      }

      if (val != 0.0) {
        K->d->data[n - 1] = val;
        K->rowidx->data[n - 1] = nb;
        n++;
      }
    }
  }

  K->colidx->data[K->colidx->size[0] - 1] = n;

  /*  Assembly of the global stiffness matrix */
  cudaFree(*gpu_dv);
  cudaFree(*gpu_dv1);
  cudaFree(*gpu_dv2);
  cudaFree(*gpu_nodes);
  cudaFree(*gpu_elements);
  cudaFree(*gpu_jK);
  cudaFree(*gpu_iK);
  cudaFree(*gpu_subs);
  cudaFree(*gpu_SZ);
  cudaFree(*gpu_idx);
  cudaFree(*gpu_ycol);
  cudaFree(*b_gpu_b_size);
  cudaFree(*gpu_b_size);
  cudaFree(*gpu_b_data);
  cudaFree(*b_gpu_b_data);
  cudaFree(*gpu_indx_size);
  cudaFree(*gpu_r_size);
  cudaFree(*gpu_sz);
  cudaFree(*gpu_idx_size);
  cudaFree(*gpu_iwork);
  cudaFree(*gpu_sortedIndices_data);
  cudaFree(*gpu_r_data);
  cudaFree(*c_gpu_b_size);
  cudaFree(*gpu_ipos);
  cudaFree(*gpu_filled_data);
  cudaFree(*gpu_Ke);
  cudaFree(*gpu_Afull_data);
  cudaFree(*gpu_ridxInt_size);
  cudaFree(*gpu_cidxInt_size);
  cudaFree(*gpu_sortedIndices_size);
  cudaFree(*gpu_tunableEnvironment);
}

/* End of code generation (StiffMas.cu) */
