#ifndef BUILDPART_KDSVM_H
#define BUILDPART_KDSVM_H

#include "svm.h"

#ifdef __cplusplus
#include <memory>
#include "psp_mcmc.h"


extern "C"
{
#endif

typedef union PSP_KdSVMTree_Data_ {
    struct svm_model* model;
    size_t pattern;
} PSP_KdSVMTree_Data;

typedef struct PSP_NodeRec_ PSP_NodeRec, *PSP_Node;
struct PSP_NodeRec_ {
    PSP_Node left;
    PSP_Node right;
};

typedef struct PSP_KdSVMTreeRec_ {
    PSP_NodeRec node;
    PSP_KdSVMTree_Data data;
} PSP_KdSVMTreeRec, *PSP_KdSVMTree;

#ifdef __cplusplus
}

struct KdSVM_Internal {
    using KdSVM_InternalPtr = std::unique_ptr<KdSVM_Internal>;
    KdSVM_InternalPtr left;
    KdSVM_InternalPtr right;
    PSP_KdSVMTree_Data data;
    svm_problem problem;

    KdSVM_Internal(KdSVM_Internal const& other) = delete;
    KdSVM_Internal(KdSVM_Internal && other) = delete;
    KdSVM_Internal & operator=(KdSVM_Internal const& other) = delete;
    KdSVM_Internal & operator=(KdSVM_Internal && other) = delete;
    ~KdSVM_Internal();
};
using KdSVM_InternalPtr = std::unique_ptr<KdSVM_Internal>;


KdSVM_InternalPtr build_kdsvm(PSP_Result data);
PSP_KdSVMTree transform_kdsvm(KdSVM_InternalPtr const& tree);
#endif

#endif

/* EOF */
