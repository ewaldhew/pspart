#ifndef BUILDPART_MCSVM_H
#define BUILDPART_MCSVM_H

#include "svm.h"

#ifdef __cplusplus
#include "common.h"
#include "psp_mcmc.h"
#include "buildpart_common.h"


extern "C"
{
#endif

typedef struct PSP_MCSVMRec_ {
    PSP_NodeRec node;
    struct svm_model* model;
} PSP_MCSVMRec, *PSP_MCSVM;

#ifdef __cplusplus
}


PSP_MCSVM build_mcsvm(PSP_Result data, svm_parameter const* param, PSP_Memory memory);
#endif

#endif

/* EOF */
