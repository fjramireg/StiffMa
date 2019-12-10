//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: StiffMas5_emxAPI.cu
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//

// Include Files
#include "StiffMas5_emxAPI.h"
#include "StiffMas5.h"
#include "StiffMas5_emxutil.h"
#include <cstdlib>

// Function Definitions

//
// Arguments    : int numDimensions
//                int *size
// Return Type  : emxArray_int32_T *
//
emxArray_int32_T *emxCreateND_int32_T(int numDimensions, int *size)
{
  emxArray_int32_T *emx;
  int numEl;
  int i;
  emxInit_int32_T(&emx, numDimensions);
  numEl = 1;
  for (i = 0; i < numDimensions; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (int *)std::calloc(static_cast<unsigned int>(numEl), sizeof(int));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : int numDimensions
//                int *size
// Return Type  : emxArray_real_T *
//
emxArray_real_T *emxCreateND_real_T(int numDimensions, int *size)
{
  emxArray_real_T *emx;
  int numEl;
  int i;
  emxInit_real_T(&emx, numDimensions);
  numEl = 1;
  for (i = 0; i < numDimensions; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (double *)std::calloc(static_cast<unsigned int>(numEl), sizeof
    (double));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : int numDimensions
//                int *size
// Return Type  : emxArray_uint32_T *
//
emxArray_uint32_T *emxCreateND_uint32_T(int numDimensions, int *size)
{
  emxArray_uint32_T *emx;
  int numEl;
  int i;
  emxInit_uint32_T(&emx, numDimensions);
  numEl = 1;
  for (i = 0; i < numDimensions; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (unsigned int *)std::calloc(static_cast<unsigned int>(numEl),
    sizeof(unsigned int));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : int *data
//                int numDimensions
//                int *size
// Return Type  : emxArray_int32_T *
//
emxArray_int32_T *emxCreateWrapperND_int32_T(int *data, int numDimensions, int
  *size)
{
  emxArray_int32_T *emx;
  int numEl;
  int i;
  emxInit_int32_T(&emx, numDimensions);
  numEl = 1;
  for (i = 0; i < numDimensions; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : double *data
//                int numDimensions
//                int *size
// Return Type  : emxArray_real_T *
//
emxArray_real_T *emxCreateWrapperND_real_T(double *data, int numDimensions, int *
  size)
{
  emxArray_real_T *emx;
  int numEl;
  int i;
  emxInit_real_T(&emx, numDimensions);
  numEl = 1;
  for (i = 0; i < numDimensions; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : unsigned int *data
//                int numDimensions
//                int *size
// Return Type  : emxArray_uint32_T *
//
emxArray_uint32_T *emxCreateWrapperND_uint32_T(unsigned int *data, int
  numDimensions, int *size)
{
  emxArray_uint32_T *emx;
  int numEl;
  int i;
  emxInit_uint32_T(&emx, numDimensions);
  numEl = 1;
  for (i = 0; i < numDimensions; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : int *data
//                int rows
//                int cols
// Return Type  : emxArray_int32_T *
//
emxArray_int32_T *emxCreateWrapper_int32_T(int *data, int rows, int cols)
{
  emxArray_int32_T *emx;
  emxInit_int32_T(&emx, 2);
  emx->size[0] = rows;
  emx->size[1] = cols;
  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = rows * cols;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : double *data
//                int rows
//                int cols
// Return Type  : emxArray_real_T *
//
emxArray_real_T *emxCreateWrapper_real_T(double *data, int rows, int cols)
{
  emxArray_real_T *emx;
  emxInit_real_T(&emx, 2);
  emx->size[0] = rows;
  emx->size[1] = cols;
  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = rows * cols;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : unsigned int *data
//                int rows
//                int cols
// Return Type  : emxArray_uint32_T *
//
emxArray_uint32_T *emxCreateWrapper_uint32_T(unsigned int *data, int rows, int
  cols)
{
  emxArray_uint32_T *emx;
  emxInit_uint32_T(&emx, 2);
  emx->size[0] = rows;
  emx->size[1] = cols;
  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = rows * cols;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : int rows
//                int cols
// Return Type  : emxArray_int32_T *
//
emxArray_int32_T *emxCreate_int32_T(int rows, int cols)
{
  emxArray_int32_T *emx;
  int numEl;
  emxInit_int32_T(&emx, 2);
  emx->size[0] = rows;
  numEl = rows * cols;
  emx->size[1] = cols;
  emx->data = (int *)std::calloc(static_cast<unsigned int>(numEl), sizeof(int));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : int rows
//                int cols
// Return Type  : emxArray_real_T *
//
emxArray_real_T *emxCreate_real_T(int rows, int cols)
{
  emxArray_real_T *emx;
  int numEl;
  emxInit_real_T(&emx, 2);
  emx->size[0] = rows;
  numEl = rows * cols;
  emx->size[1] = cols;
  emx->data = (double *)std::calloc(static_cast<unsigned int>(numEl), sizeof
    (double));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : int rows
//                int cols
// Return Type  : emxArray_uint32_T *
//
emxArray_uint32_T *emxCreate_uint32_T(int rows, int cols)
{
  emxArray_uint32_T *emx;
  int numEl;
  emxInit_uint32_T(&emx, 2);
  emx->size[0] = rows;
  numEl = rows * cols;
  emx->size[1] = cols;
  emx->data = (unsigned int *)std::calloc(static_cast<unsigned int>(numEl),
    sizeof(unsigned int));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : emxArray_int32_T *emxArray
// Return Type  : void
//
void emxDestroyArray_int32_T(emxArray_int32_T *emxArray)
{
  emxFree_int32_T(&emxArray);
}

//
// Arguments    : emxArray_real_T *emxArray
// Return Type  : void
//
void emxDestroyArray_real_T(emxArray_real_T *emxArray)
{
  emxFree_real_T(&emxArray);
}

//
// Arguments    : emxArray_uint32_T *emxArray
// Return Type  : void
//
void emxDestroyArray_uint32_T(emxArray_uint32_T *emxArray)
{
  emxFree_uint32_T(&emxArray);
}

//
// Arguments    : coder_internal_sparse emxArray
// Return Type  : void
//
void emxDestroy_coder_internal_sparse(coder_internal_sparse emxArray)
{
  c_emxFreeStruct_coder_internal_(&emxArray);
}

//
// Arguments    : emxArray_real_T **pEmxArray
//                int numDimensions
// Return Type  : void
//
void emxInitArray_real_T(emxArray_real_T **pEmxArray, int numDimensions)
{
  emxInit_real_T(pEmxArray, numDimensions);
}

//
// Arguments    : emxArray_uint32_T **pEmxArray
//                int numDimensions
// Return Type  : void
//
void emxInitArray_uint32_T(emxArray_uint32_T **pEmxArray, int numDimensions)
{
  emxInit_uint32_T(pEmxArray, numDimensions);
}

//
// Arguments    : coder_internal_sparse *pStruct
// Return Type  : void
//
void emxInit_coder_internal_sparse(coder_internal_sparse *pStruct)
{
  c_emxInitStruct_coder_internal_(pStruct);
}

//
// File trailer for StiffMas5_emxAPI.cu
//
// [EOF]
//
