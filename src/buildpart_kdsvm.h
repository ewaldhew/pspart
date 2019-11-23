#ifndef BUILDPART_KDSVM_H
#define BUILDPART_KDSVM_H

#include "svm.h"

#ifdef __cplusplus
#include "common.h"
#include "psp_mcmc.h"
#include "buildpart_common.h"


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


PSP_KdSVMTree build_kdsvm(PSP_Result data, svm_parameter const* param, PSP_Memory memory);
#endif

#endif

/* EOF */
