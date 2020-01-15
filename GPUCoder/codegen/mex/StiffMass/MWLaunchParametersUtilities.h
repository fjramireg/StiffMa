/* Copyright 2017-2019 The MathWorks, Inc. */

#ifdef __CUDACC__
#ifndef MW_LAUNCH_PARAMETERS_HPP
#define MW_LAUNCH_PARAMETERS_HPP

#define MW_LAUNCH_UNSIGNED_TYPE unsigned long long
#define MW_HOST_DEVICE __host__ __device__

MW_HOST_DEVICE bool mwGetLaunchParameters(double numberOfThreads,
                                          dim3* grid,
                                          dim3* block,
                                          MW_LAUNCH_UNSIGNED_TYPE MAX_THREADS_PER_BLOCK,
                                          MW_LAUNCH_UNSIGNED_TYPE MAX_BLOCKS_PER_GRID);

MW_HOST_DEVICE bool mwGetLaunchParameters1D(double numberOfThreads,
                                            dim3* grid,
                                            dim3* block,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_THREADS_PER_BLOCK,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_BLOCKS_PER_GRID);

MW_HOST_DEVICE bool mwGetLaunchParameters2D(double numberOfThreads,
                                            dim3* grid,
                                            dim3* block,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_THREADS_PER_BLOCK,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_BLOCKS_PER_GRID);

MW_HOST_DEVICE bool mwGetLaunchParameters3D(double numberOfThreads,
                                            dim3* grid,
                                            dim3* block,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_THREADS_PER_BLOCK,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_BLOCKS_PER_GRID);

MW_HOST_DEVICE bool mwApplyLaunchParameters(double numberOfThreads,
                                            const dim3* ingrid,
                                            const dim3* inblock,
                                            dim3* outgrid,
                                            dim3* outblock,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_THREADS_PER_BLOCK,
                                            MW_LAUNCH_UNSIGNED_TYPE MAX_BLOCKS_PER_GRID);

MW_HOST_DEVICE void mwWrapToMultipleOf32(dim3* in);

MW_HOST_DEVICE bool mwWithBlockSizeLimits(const dim3* in);

MW_HOST_DEVICE double mwTotalThreads(const dim3* obj);

MW_HOST_DEVICE MW_LAUNCH_UNSIGNED_TYPE mwRoundToMultipleOf32(MW_LAUNCH_UNSIGNED_TYPE val);

MW_HOST_DEVICE bool mwValidDim3(const dim3* obj);

MW_HOST_DEVICE void mwResetDim3(dim3* obj);

MW_HOST_DEVICE void mwResetDim3ToZeros(dim3* obj);

#endif
#endif
