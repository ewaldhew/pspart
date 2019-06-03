#ifndef BUILDPART_KDSVM_H
#define BUILDPART_KDSVM_H

#include "svm.h"

#ifdef __cplusplus
#include "common.h"
#include "psp_mcmc.h"


extern "C"
{
#endif

typedef union PSP_KdSVMTree_Data_ {
    struct svm_model* model;
    size_t pattern;
} PSP_KdSVMTree_Data;

typedef struct PSP_KdSVMTreeRec_ {
    PSP_NodeRec node;
    PSP_KdSVMTree_Data data;
} PSP_KdSVMTreeRec, *PSP_KdSVMTree;

#ifdef __cplusplus
}


typedef struct PSP_MemoryRec *PSP_Memory;
PSP_KdSVMTree build_kdsvm(PSP_Result data, PSP_Memory memory);
#endif

#endif

/* EOF */