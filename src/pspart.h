#ifndef PSPART_H
#define PSPART_H

#include <stddef.h>
#include <errno.h>

#define ERR_UNHANDLED_EXCEPTION -1

#include "buildpart.h"
#include "psp_mcmc.h"


typedef long Fixed;

typedef struct PSP_Handle_
#ifdef __cplusplus
{
    size_t n_dim;
    PSP_Result psp_regions;
    KdSVM_InternalPtr kdsvm;
}
#endif
    *PSP_Handle;


#ifdef __cplusplus
extern "C"
{
#endif

typedef size_t (*Sampling_Func)(void* sampling_context,
                                Fixed* point);

typedef struct PSP_Sampling_CallbackRec_ {
    void* sampling_context;
    Sampling_Func sampler;
} PSP_Sampling_CallbackRec, *PSP_Sampling_Callback;


PSP_Handle PSP_New(size_t dim);
void PSP_Close(PSP_Handle handle);

int PSP_Get_Regions(PSP_Handle handle,
                    PSP_Sampling_Callback sampling_callback,
                    int num_start_points,
                    Fixed *start_points,
                    Fixed *min_coords,
                    Fixed *max_coords,
                    PSP_Options options,
                    PSP_Result_Mode result_mode);

int PSP_Build_Partition_KdSVM(PSP_Handle handle,
                              PSP_KdSVMTree* tree);

#ifdef __cplusplus
}
#endif


#endif

/* EOF */
