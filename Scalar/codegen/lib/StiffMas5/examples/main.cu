//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: main.cu
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//

//***********************************************************************
// This automatically generated example CUDA main file shows how to call
// entry-point functions that MATLAB Coder generated. You must customize
// this file for your application. Do not modify this file directly.
// Instead, make a copy of this file, modify it, and integrate it into
// your development environment.
//
// This file initializes entry-point function arguments to a default
// size and value before calling the entry-point functions. It does
// not store or use any values returned from the entry-point functions.
// If necessary, it does pre-allocate memory for returned values.
// You can use this file as a starting point for a main function that
// you can deploy in your application.
//
// After you copy the file, and before you deploy it, you must make the
// following changes:
// * For variable-size function arguments, change the example sizes to
// the sizes that your application requires.
// * Change the example values of function arguments to the values that
// your application requires.
// * If the entry-point functions return values, store these values or
// otherwise use them as required by your application.
//
//***********************************************************************

// Include Files
#include "main.h"
#include "StiffMas5.h"
#include "StiffMas5_emxAPI.h"
#include "StiffMas5_terminate.h"

// Function Declarations
static emxArray_real_T *argInit_Unboundedx3_real_T();
static emxArray_uint32_T *argInit_Unboundedx8_uint32_T();
static double argInit_real_T();
static unsigned int argInit_uint32_T();
static void main_StiffMas5();

// Function Definitions

//
// Arguments    : void
// Return Type  : emxArray_real_T *
//
static emxArray_real_T *argInit_Unboundedx3_real_T()
{
  emxArray_real_T *result;
  int loopUpperBound;
  int idx0;
  int idx1;

  // Set the size of the array.
  // Change this size to the value that the application requires.
  result = emxCreate_real_T(2, 3);

  // Loop over the array to initialize each element.
  loopUpperBound = result->size[0U];
  for (idx0 = 0; idx0 < loopUpperBound; idx0++) {
    for (idx1 = 0; idx1 < 3; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result->data[idx0 + result->size[0] * idx1] = argInit_real_T();
    }
  }

  return result;
}

//
// Arguments    : void
// Return Type  : emxArray_uint32_T *
//
static emxArray_uint32_T *argInit_Unboundedx8_uint32_T()
{
  emxArray_uint32_T *result;
  int loopUpperBound;
  int idx0;
  int idx1;

  // Set the size of the array.
  // Change this size to the value that the application requires.
  result = emxCreate_uint32_T(2, 8);

  // Loop over the array to initialize each element.
  loopUpperBound = result->size[0U];
  for (idx0 = 0; idx0 < loopUpperBound; idx0++) {
    for (idx1 = 0; idx1 < 8; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result->data[idx0 + result->size[0] * idx1] = argInit_uint32_T();
    }
  }

  return result;
}

//
// Arguments    : void
// Return Type  : double
//
static double argInit_real_T()
{
  return 0.0;
}

//
// Arguments    : void
// Return Type  : unsigned int
//
static unsigned int argInit_uint32_T()
{
  return 0U;
}

//
// Arguments    : void
// Return Type  : void
//
static void main_StiffMas5()
{
  coder_internal_sparse K;
  emxArray_uint32_T *elements;
  emxArray_real_T *nodes;
  emxInit_coder_internal_sparse(&K);

  // Initialize function 'StiffMas5' input arguments.
  // Initialize function input argument 'elements'.
  elements = argInit_Unboundedx8_uint32_T();

  // Initialize function input argument 'nodes'.
  nodes = argInit_Unboundedx3_real_T();

  // Call the entry-point 'StiffMas5'.
  StiffMas5(elements, nodes, argInit_real_T(), &K);
  emxDestroy_coder_internal_sparse(K);
  emxDestroyArray_real_T(nodes);
  emxDestroyArray_uint32_T(elements);
}

//
// Arguments    : int argc
//                const char * const argv[]
// Return Type  : int
//
int main(int, const char * const [])
{
  // The initialize function is being called automatically from your entry-point function. So, a call to initialize is not included here. 
  // Invoke the entry-point functions.
  // You can call entry-point functions multiple times.
  main_StiffMas5();

  // Terminate the application.
  // You do not need to do this more than one time.
  StiffMas5_terminate();
  return 0;
}

//
// File trailer for main.cu
//
// [EOF]
//
