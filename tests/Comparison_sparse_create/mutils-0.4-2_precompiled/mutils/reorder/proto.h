/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * proto.h
 *
 * This file contains header files
 *
 * Started 10/19/95
 * George
 *
 * $Id: proto.h,v 1.1 1998/11/27 17:59:28 karypis Exp $
 *
 */

/* balance.c */
void Balance2Way(CtrlType *, GraphType *, long *, float);
void Bnd2WayBalance(CtrlType *, GraphType *, long *);
void General2WayBalance(CtrlType *, GraphType *, long *);

/* bucketsort.c */
void BucketSortKeysInc(long, long, idxtype *, idxtype *, idxtype *);

/* ccgraph.c */
void CreateCoarseGraph(CtrlType *, GraphType *, long, idxtype *, idxtype *);
void CreateCoarseGraphNoMask(CtrlType *, GraphType *, long, idxtype *, idxtype *);
void CreateCoarseGraph_NVW(CtrlType *, GraphType *, long, idxtype *, idxtype *);
GraphType *SetUpCoarseGraph(GraphType *, long, long);
void ReAdjustMemory(GraphType *, GraphType *, long);

/* coarsen.c */
GraphType *Coarsen2Way(CtrlType *, GraphType *);

/* compress.c */
void CompressGraph(CtrlType *, GraphType *, long, idxtype *, idxtype *, idxtype *, idxtype *);
void PruneGraph(CtrlType *, GraphType *, long, idxtype *, idxtype *, idxtype *, float);

/* debug.c */
long ComputeCut(GraphType *, idxtype *);
long CheckBnd(GraphType *);
long CheckBnd2(GraphType *);
long CheckNodeBnd(GraphType *, long);
long CheckRInfo(RInfoType *);
long CheckNodePartitionParams(GraphType *);
long IsSeparable(GraphType *);

/* estmem.c */
void METIS_EstimateMemory(long *, idxtype *, idxtype *, long *, long *, long *);
void EstimateCFraction(long, idxtype *, idxtype *, float *, float *);
long ComputeCoarseGraphSize(long, idxtype *, idxtype *, long, idxtype *, idxtype *, idxtype *);

/* fm.c */
void FM_2WayEdgeRefine(CtrlType *, GraphType *, long *, long);

/* fortran.c */
void Change2CNumbering(long, idxtype *, idxtype *);
void Change2FNumbering(long, idxtype *, idxtype *, idxtype *);
void Change2FNumbering2(long, idxtype *, idxtype *);
void Change2FNumberingOrder(long, idxtype *, idxtype *, idxtype *, idxtype *);
void ChangeMesh2CNumbering(long, idxtype *);
void ChangeMesh2FNumbering(long, idxtype *, long, idxtype *, idxtype *);
void ChangeMesh2FNumbering2(long, idxtype *, long, long, idxtype *, idxtype *);

/* frename.c */
void METIS_PARTGRAPHRECURSIVE(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void metis_partgraphrecursive(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void metis_partgraphrecursive_(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void metis_partgraphrecursive__(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void METIS_WPARTGRAPHRECURSIVE(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void metis_wpartgraphrecursive(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void metis_wpartgraphrecursive_(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void metis_wpartgraphrecursive__(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void METIS_PARTGRAPHKWAY(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void metis_partgraphkway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void metis_partgraphkway_(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void metis_partgraphkway__(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void METIS_WPARTGRAPHKWAY(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void metis_wpartgraphkway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void metis_wpartgraphkway_(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void metis_wpartgraphkway__(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void METIS_EDGEND(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_edgend(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_edgend_(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_edgend__(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void METIS_NODEND(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_nodend(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_nodend_(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_nodend__(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void METIS_NODEWND(long *, idxtype *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_nodewnd(long *, idxtype *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_nodewnd_(long *, idxtype *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void metis_nodewnd__(long *, idxtype *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void METIS_PARTMESHNODAL(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void metis_partmeshnodal(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void metis_partmeshnodal_(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void metis_partmeshnodal__(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void METIS_PARTMESHDUAL(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void metis_partmeshdual(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void metis_partmeshdual_(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void metis_partmeshdual__(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void METIS_MESHTONODAL(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void metis_meshtonodal(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void metis_meshtonodal_(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void metis_meshtonodal__(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void METIS_MESHTODUAL(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void metis_meshtodual(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void metis_meshtodual_(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void metis_meshtodual__(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void METIS_ESTIMATEMEMORY(long *, idxtype *, idxtype *, long *, long *, long *);
void metis_estimatememory(long *, idxtype *, idxtype *, long *, long *, long *);
void metis_estimatememory_(long *, idxtype *, idxtype *, long *, long *, long *);
void metis_estimatememory__(long *, idxtype *, idxtype *, long *, long *, long *);
void METIS_MCPARTGRAPHRECURSIVE(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void metis_mcpartgraphrecursive(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void metis_mcpartgraphrecursive_(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void metis_mcpartgraphrecursive__(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void METIS_MCPARTGRAPHKWAY(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void metis_mcpartgraphkway(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void metis_mcpartgraphkway_(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void metis_mcpartgraphkway__(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void METIS_PARTGRAPHVKWAY(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void metis_partgraphvkway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void metis_partgraphvkway_(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void metis_partgraphvkway__(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void METIS_WPARTGRAPHVKWAY(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void metis_wpartgraphvkway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void metis_wpartgraphvkway_(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void metis_wpartgraphvkway__(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);

/* graph.c */
void SetUpGraph(GraphType *, long, long, long, idxtype *, idxtype *, idxtype *, idxtype *, long);
void SetUpGraphKway(GraphType *, long, idxtype *, idxtype *);
void SetUpGraph2(GraphType *, long, long, idxtype *, idxtype *, float *, idxtype *);
void VolSetUpGraph(GraphType *, long, long, long, idxtype *, idxtype *, idxtype *, idxtype *, long);
void RandomizeGraph(GraphType *);
long IsConnectedSubdomain(CtrlType *, GraphType *, long, long);
long IsConnected(CtrlType *, GraphType *, long);
long IsConnected2(GraphType *, long);
long FindComponents(CtrlType *, GraphType *, idxtype *, idxtype *);

/* initpart.c */
void Init2WayPartition(CtrlType *, GraphType *, long *, float);
void InitSeparator(CtrlType *, GraphType *, float);
void GrowBisection(CtrlType *, GraphType *, long *, float);
void GrowBisectionNode(CtrlType *, GraphType *, float);
void RandomBisection(CtrlType *, GraphType *, long *, float);

/* kmetis.c */
void METIS_PartGraphKway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void METIS_WPartGraphKway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
long MlevelKWayPartitioning(CtrlType *, GraphType *, long, idxtype *, float *, float);

/* kvmetis.c */
void METIS_PartGraphVKway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void METIS_WPartGraphVKway(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
long MlevelVolKWayPartitioning(CtrlType *, GraphType *, long, idxtype *, float *, float);

/* kwayfm.c */
void Random_KWayEdgeRefine(CtrlType *, GraphType *, long, float *, float, long, long);
void Greedy_KWayEdgeRefine(CtrlType *, GraphType *, long, float *, float, long);
void Greedy_KWayEdgeBalance(CtrlType *, GraphType *, long, float *, float, long);

/* kwayrefine.c */
void RefineKWay(CtrlType *, GraphType *, GraphType *, long, float *, float);
void AllocateKWayPartitionMemory(CtrlType *, GraphType *, long);
void ComputeKWayPartitionParams(CtrlType *, GraphType *, long);
void ProjectKWayPartition(CtrlType *, GraphType *, long);
long IsBalanced(idxtype *, long, float *, float);
void ComputeKWayBoundary(CtrlType *, GraphType *, long);
void ComputeKWayBalanceBoundary(CtrlType *, GraphType *, long);

/* kwayvolfm.c */
void Random_KWayVolRefine(CtrlType *, GraphType *, long, float *, float, long, long);
void Random_KWayVolRefineMConn(CtrlType *, GraphType *, long, float *, float, long, long);
void Greedy_KWayVolBalance(CtrlType *, GraphType *, long, float *, float, long);
void Greedy_KWayVolBalanceMConn(CtrlType *, GraphType *, long, float *, float, long);
void KWayVolUpdate(CtrlType *, GraphType *, long, long, long, idxtype *, idxtype *, idxtype *);
void ComputeKWayVolume(GraphType *, long, idxtype *, idxtype *, idxtype *);
long ComputeVolume(GraphType *, idxtype *);
void CheckVolKWayPartitionParams(CtrlType *, GraphType *, long);
void ComputeVolSubDomainGraph(GraphType *, long, idxtype *, idxtype *);
void EliminateVolSubDomainEdges(CtrlType *, GraphType *, long, float *);
void EliminateVolComponents(CtrlType *, GraphType *, long, float *, float);

/* kwayvolrefine.c */
void RefineVolKWay(CtrlType *, GraphType *, GraphType *, long, float *, float);
void AllocateVolKWayPartitionMemory(CtrlType *, GraphType *, long);
void ComputeVolKWayPartitionParams(CtrlType *, GraphType *, long);
void ComputeKWayVolGains(CtrlType *, GraphType *, long);
void ProjectVolKWayPartition(CtrlType *, GraphType *, long);
void ComputeVolKWayBoundary(CtrlType *, GraphType *, long);
void ComputeVolKWayBalanceBoundary(CtrlType *, GraphType *, long);

/* match.c */
void Match_RM(CtrlType *, GraphType *);
void Match_RM_NVW(CtrlType *, GraphType *);
void Match_HEM(CtrlType *, GraphType *);
void Match_SHEM(CtrlType *, GraphType *);

/* mbalance.c */
void MocBalance2Way(CtrlType *, GraphType *, float *, float);
void MocGeneral2WayBalance(CtrlType *, GraphType *, float *, float);

/* mbalance2.c */
void MocBalance2Way2(CtrlType *, GraphType *, float *, float *);
void MocGeneral2WayBalance2(CtrlType *, GraphType *, float *, float *);
void SelectQueue3(long, float *, float *, long *, long *, PQueueType [MAXNCON][2], float *);

/* mcoarsen.c */
GraphType *MCCoarsen2Way(CtrlType *, GraphType *);

/* memory.c */
void AllocateWorkSpace(CtrlType *, GraphType *, long);
void FreeWorkSpace(CtrlType *, GraphType *);
long WspaceAvail(CtrlType *);
idxtype *idxwspacemalloc(CtrlType *, long);
void idxwspacefree(CtrlType *, long);
float *fwspacemalloc(CtrlType *, long);
void fwspacefree(CtrlType *, long);
GraphType *CreateGraph(void);
void InitGraph(GraphType *);
void FreeGraph(GraphType *);

/* mesh.c */
void METIS_MeshToDual(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void METIS_MeshToNodal(long *, long *, idxtype *, long *, long *, idxtype *, idxtype *);
void GENDUALMETIS(long, long, long, idxtype *, idxtype *, idxtype *adjncy);
void TRINODALMETIS(long, long, idxtype *, idxtype *, idxtype *adjncy);
void TETNODALMETIS(long, long, idxtype *, idxtype *, idxtype *adjncy);
void HEXNODALMETIS(long, long, idxtype *, idxtype *, idxtype *adjncy);
void QUADNODALMETIS(long, long, idxtype *, idxtype *, idxtype *adjncy);

/* meshpart.c */
void METIS_PartMeshNodal(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);
void METIS_PartMeshDual(long *, long *, idxtype *, long *, long *, long *, long *, idxtype *, idxtype *);

/* mfm.c */
void MocFM_2WayEdgeRefine(CtrlType *, GraphType *, float *, long);
void SelectQueue(long, float *, float *, long *, long *, PQueueType [MAXNCON][2]);
long BetterBalance(long, float *, float *, float *);
float Compute2WayHLoadImbalance(long, float *, float *);
void Compute2WayHLoadImbalanceVec(long, float *, float *, float *);

/* mfm2.c */
void MocFM_2WayEdgeRefine2(CtrlType *, GraphType *, float *, float *, long);
void SelectQueue2(long, float *, float *, long *, long *, PQueueType [MAXNCON][2], float *);
long IsBetter2wayBalance(long, float *, float *, float *);

/* mincover.o */
void MinCover(idxtype *, idxtype *, long, long, idxtype *, long *);
long MinCover_Augment(idxtype *, idxtype *, long, idxtype *, idxtype *, idxtype *, long);
void MinCover_Decompose(idxtype *, idxtype *, long, long, idxtype *, idxtype *, long *);
void MinCover_ColDFS(idxtype *, idxtype *, long, idxtype *, idxtype *, long);
void MinCover_RowDFS(idxtype *, idxtype *, long, idxtype *, idxtype *, long);

/* minitpart.c */
void MocInit2WayPartition(CtrlType *, GraphType *, float *, float);
void MocGrowBisection(CtrlType *, GraphType *, float *, float);
void MocRandomBisection(CtrlType *, GraphType *, float *, float);
void MocInit2WayBalance(CtrlType *, GraphType *, float *);
long SelectQueueoneWay(long, float *, float *, long, PQueueType [MAXNCON][2]);

/* minitpart2.c */
void MocInit2WayPartition2(CtrlType *, GraphType *, float *, float *);
void MocGrowBisection2(CtrlType *, GraphType *, float *, float *);
void MocGrowBisectionNew2(CtrlType *, GraphType *, float *, float *);
void MocInit2WayBalance2(CtrlType *, GraphType *, float *, float *);
long SelectQueueOneWay2(long, float *, PQueueType [MAXNCON][2], float *);

/* mkmetis.c */
void METIS_mCPartGraphKway(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
long MCMlevelKWayPartitioning(CtrlType *, GraphType *, long, idxtype *, float *);

/* mkwayfmh.c */
void MCRandom_KWayEdgeRefineHorizontal(CtrlType *, GraphType *, long, float *, long);
void MCGreedy_KWayEdgeBalanceHorizontal(CtrlType *, GraphType *, long, float *, long);
long AreAllHVwgtsBelow(long, float, float *, float, float *, float *);
long AreAllHVwgtsAbove(long, float, float *, float, float *, float *);
void ComputeHKWayLoadImbalance(long, long, float *, float *);
long MocIsHBalanced(long, long, float *, float *);
long IsHBalanceBetterFT(long, long, float *, float *, float *, float *);
long IsHBalanceBetterTT(long, long, float *, float *, float *, float *);

/* mkwayrefine.c */
void MocRefineKWayHorizontal(CtrlType *, GraphType *, GraphType *, long, float *);
void MocAllocateKWayPartitionMemory(CtrlType *, GraphType *, long);
void MocComputeKWayPartitionParams(CtrlType *, GraphType *, long);
void MocProjectKWayPartition(CtrlType *, GraphType *, long);
void MocComputeKWayBalanceBoundary(CtrlType *, GraphType *, long);

/* mmatch.c */
void MCMatch_RM(CtrlType *, GraphType *);
void MCMatch_HEM(CtrlType *, GraphType *);
void MCMatch_SHEM(CtrlType *, GraphType *);
void MCMatch_SHEBM(CtrlType *, GraphType *, long);
void MCMatch_SBHEM(CtrlType *, GraphType *, long);
float BetterVBalance(long, long, float *, float *, float *);
long AreAllVwgtsBelowFast(long, float *, float *, float);

/* mmd.c */
void genmmd(long, idxtype *, idxtype *, idxtype *, idxtype *, long , idxtype *, idxtype *, idxtype *, idxtype *, long, long *);
void mmdelm(long, idxtype *xadj, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, long, long);
long  mmdlong(long, idxtype *xadj, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *);
void mmdnum(long, idxtype *, idxtype *, idxtype *);
void mmdupd(long, long, idxtype *, idxtype *, long, long *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, long, long *tag);

/* mpmetis.c */
void METIS_mCPartGraphRecursive(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *);
void METIS_mCHPartGraphRecursive(long *, long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *);
void METIS_mCPartGraphRecursivelongernal(long *, long *, idxtype *, idxtype *, float *, idxtype *, long *, long *, long *, idxtype *);
void METIS_mCHPartGraphRecursivelongernal(long *, long *, idxtype *, idxtype *, float *, idxtype *, long *, float *, long *, long *, idxtype *);
long MCMlevelRecursiveBisection(CtrlType *, GraphType *, long, idxtype *, float, long);
long MCHMlevelRecursiveBisection(CtrlType *, GraphType *, long, idxtype *, float *, long);
void MCMlevelEdgeBisection(CtrlType *, GraphType *, float *, float);
void MCHMlevelEdgeBisection(CtrlType *, GraphType *, float *, float *);

/* mrefine.c */
void MocRefine2Way(CtrlType *, GraphType *, GraphType *, float *, float);
void MocAllocate2WayPartitionMemory(CtrlType *, GraphType *);
void MocCompute2WayPartitionParams(CtrlType *, GraphType *);
void MocProject2WayPartition(CtrlType *, GraphType *);

/* mrefine2.c */
void MocRefine2Way2(CtrlType *, GraphType *, GraphType *, float *, float *);

/* mutil.c */
long AreAllVwgtsBelow(long, float, float *, float, float *, float);
long AreAnyVwgtsBelow(long, float, float *, float, float *, float);
long AreAllVwgtsAbove(long, float, float *, float, float *, float);
float ComputeLoadImbalance(long, long, float *, float *);
long AreAllBelow(long, float *, float *);

/* myqsort.c */
void iidxsort(long, idxtype *);
void ilongsort(long, long *);
void ikeysort(long, KeyValueType *);
void ikeyvalsort(long, KeyValueType *);

/* ometis.c */
void METIS_EdgeND(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void METIS_NodeND(long *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void METIS_NodeWND(long *, idxtype *, idxtype *, idxtype *, long *, long *, idxtype *, idxtype *); 
void MlevelNestedDissection(CtrlType *, GraphType *, idxtype *, float, long);
void MlevelNestedDissectionCC(CtrlType *, GraphType *, idxtype *, float, long);
void MlevelNodeBisectionMultiple(CtrlType *, GraphType *, long *, float);
void MlevelNodeBisection(CtrlType *, GraphType *, long *, float);
void SplitGraphOrder(CtrlType *, GraphType *, GraphType *, GraphType *);
void MMDOrder(CtrlType *, GraphType *, idxtype *, long);
long SplitGraphOrderCC(CtrlType *, GraphType *, GraphType *, long, idxtype *, idxtype *);

/* parmetis.c */
void METIS_PartGraphKway2(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void METIS_WPartGraphKway2(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
void METIS_NodeNDP(long, idxtype *, idxtype *, long, long *, idxtype *, idxtype *, idxtype *);
void MlevelNestedDissectionP(CtrlType *, GraphType *, idxtype *, long, long, long, idxtype *);
void METIS_NodeComputeSeparator(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, idxtype *); 
void METIS_EdgeComputeSeparator(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, idxtype *); 

/* pmetis.c */
void METIS_PartGraphRecursive(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, long *, long *, idxtype *); 
void METIS_WPartGraphRecursive(long *, idxtype *, idxtype *, idxtype *, idxtype *, long *, long *, long *, float *, long *, long *, idxtype *); 
long MlevelRecursiveBisection(CtrlType *, GraphType *, long, idxtype *, float *, float, long);
void MlevelEdgeBisection(CtrlType *, GraphType *, long *, float);
void SplitGraphPart(CtrlType *, GraphType *, GraphType *, GraphType *);
void SetUpSplitGraph(GraphType *, GraphType *, long, long);

/* pqueue.c */
void PQueueInit(CtrlType *ctrl, PQueueType *, long, long);
void PQueueReset(PQueueType *);
void PQueueFree(CtrlType *ctrl, PQueueType *);
long PQueueGetSize(PQueueType *);
long PQueueInsert(PQueueType *, long, long);
long PQueueDelete(PQueueType *, long, long);
long PQueueUpdate(PQueueType *, long, long, long);
void PQueueUpdateUp(PQueueType *, long, long, long);
long PQueueGetMax(PQueueType *);
long PQueueSeeMax(PQueueType *);
long PQueueGetKey(PQueueType *);
long CheckHeap(PQueueType *);

/* refine.c */
void Refine2Way(CtrlType *, GraphType *, GraphType *, long *, float ubfactor);
void Allocate2WayPartitionMemory(CtrlType *, GraphType *);
void Compute2WayPartitionParams(CtrlType *, GraphType *);
void Project2WayPartition(CtrlType *, GraphType *);

/* separator.c */
void ConstructSeparator(CtrlType *, GraphType *, float);
void ConstructMinCoverSeparator0(CtrlType *, GraphType *, float);
void ConstructMinCoverSeparator(CtrlType *, GraphType *, float);

/* sfm.c */
void FM_2WayNodeRefine(CtrlType *, GraphType *, float, long);
void FM_2WayNodeRefineEqWgt(CtrlType *, GraphType *, long);
void FM_2WayNodeRefine_OneSided(CtrlType *, GraphType *, float, long);
void FM_2WayNodeBalance(CtrlType *, GraphType *, float);
long ComputeMaxNodeGain(long, idxtype *, idxtype *, idxtype *);

/* srefine.c */
void Refine2WayNode(CtrlType *, GraphType *, GraphType *, float);
void Allocate2WayNodePartitionMemory(CtrlType *, GraphType *);
void Compute2WayNodePartitionParams(CtrlType *, GraphType *);
void Project2WayNodePartition(CtrlType *, GraphType *);

/* stat.c */
void ComputePartitionInfo(GraphType *, long, idxtype *);
void ComputePartitionInfoBipartite(GraphType *, long, idxtype *);
void ComputePartitionBalance(GraphType *, long, idxtype *, float *);
float ComputeElementBalance(long, long, idxtype *);

/* subdomains.c */
void Random_KWayEdgeRefineMConn(CtrlType *, GraphType *, long, float *, float, long, long);
void Greedy_KWayEdgeBalanceMConn(CtrlType *, GraphType *, long, float *, float, long);
void PrlongSubDomainGraph(GraphType *, long, idxtype *);
void ComputeSubDomainGraph(GraphType *, long, idxtype *, idxtype *);
void EliminateSubDomainEdges(CtrlType *, GraphType *, long, float *);
void MoveGroupMConn(CtrlType *, GraphType *, idxtype *, idxtype *, long, long, long, idxtype *);
void EliminateComponents(CtrlType *, GraphType *, long, float *, float);
void MoveGroup(CtrlType *, GraphType *, long, long, long, idxtype *, idxtype *);

/* timing.c */
void InitTimers(CtrlType *);
void PrlongTimers(CtrlType *);
double seconds(void);

/* util.c */
void errexit(char *,...);
#ifndef DMALLOC
long *imalloc(long, char *);
idxtype *idxmalloc(long, char *);
float *fmalloc(long, char *);
long *ismalloc(long, long, char *);
idxtype *idxsmalloc(long, idxtype, char *);
void *GKmalloc(long, char *);
#endif
/*void GKfree(void **,...); */
long *iset(long n, long val, long *x);
idxtype *idxset(long n, idxtype val, idxtype *x);
float *sset(long n, float val, float *x);
long iamax(long, long *);
long idxamax(long, idxtype *);
long idxamax_strd(long, idxtype *, long);
long samax(long, float *);
long samax2(long, float *);
long idxamin(long, idxtype *);
long samin(long, float *);
long idxsum(long, idxtype *);
long idxsum_strd(long, idxtype *, long);
void idxadd(long, idxtype *, idxtype *);
long charsum(long, char *);
long isum(long, long *);
float ssum(long, float *);
float ssum_strd(long n, float *x, long);
void sscale(long n, float, float *x);
float snorm2(long, float *);
float sdot(long n, float *, float *);
void saxpy(long, float, float *, long, float *, long);
void RandomPermute(long, idxtype *, long);
double drand48();
void srand48(long);
long ispow2(long);
void InitRandom(long);
long ilog2(long);










/***************************************************************
* Programs Directory
****************************************************************/

/* io.c */
void ReadGraph(GraphType *, char *, long *);
void WritePartition(char *, idxtype *, long, long);
void WriteMeshPartition(char *, long, long, idxtype *, long, idxtype *);
void WritePermutation(char *, idxtype *, long);
long CheckGraph(GraphType *);
idxtype *ReadMesh(char *, long *, long *, long *);
void WriteGraph(char *, long, idxtype *, idxtype *);

/* smbfactor.c */
void ComputeFillIn(GraphType *, idxtype *);
idxtype ComputeFillIn2(GraphType *, idxtype *);
long smbfct(long, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, long *, idxtype *, idxtype *, long *);


/***************************************************************
* Test Directory
****************************************************************/
void Test_PartGraph(long, idxtype *, idxtype *);
long VerifyPart(long, idxtype *, idxtype *, idxtype *, idxtype *, long, long, idxtype *);
long VerifyWPart(long, idxtype *, idxtype *, idxtype *, idxtype *, long, float *, long, idxtype *);
void Test_PartGraphV(long, idxtype *, idxtype *);
long VerifyPartV(long, idxtype *, idxtype *, idxtype *, idxtype *, long, long, idxtype *);
long VerifyWPartV(long, idxtype *, idxtype *, idxtype *, idxtype *, long, float *, long, idxtype *);
void Test_PartGraphmC(long, idxtype *, idxtype *);
long VerifyPartmC(long, long, idxtype *, idxtype *, idxtype *, idxtype *, long, float *, long, idxtype *);
void Test_ND(long, idxtype *, idxtype *);
long VerifyND(long, idxtype *, idxtype *);


/* additional - patch */
/*
void PrintTimers(CtrlType *ctrl);
void METIS_mCPartGraphRecursiveInternal(long *nvtxs, long *ncon, idxtype *xadj, idxtype *adjncy, float *nvwgt, idxtype *adjwgt, long *nparts, long *options, long *edgecut, idxtype *part);
void METIS_mCHPartGraphRecursiveInternal(long *nvtxs, long *ncon, idxtype *xadj, idxtype *adjncy, float *nvwgt, idxtype *adjwgt, long *nparts, float *ubvec, long *options, long *edgecut, idxtype *part);
void GKfree(void **ptr1,...);
*/