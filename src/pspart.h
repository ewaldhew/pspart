#ifndef PSPART_H
#define PSPART_H

#include <stddef.h>
#include <errno.h>

#define ERR_UNHANDLED_EXCEPTION -1

#include "common.h"
#include "buildpart.h"
#include "psp_mcmc.h"


typedef long Fixed;

typedef struct PSP_Handle_ *PSP_Handle;


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


/** Allocates a new instance of PSP */
PSP_Handle PSP_New(size_t dim);
/** Deallocates the PSP instance */
void PSP_Close(PSP_Handle handle);

/**
 * Discovers parameter regions with the given space and model.
 *
 * - sampling_callback: A closure object representing the model. The function
 *     should accept a point and return a number representing the data pattern.
 *
 * - points: Lists of coordinates in 16-bit fixed point format.
 *
 * - options: Options for the PSP Monte Carlo algorithm.
 *     - maxPsp: Maximum number of search cycles allowed before termination. The
 *       search stops when the number of search cycles of every Markov chain (in
 *       every discovered region) reaches `maxPSP`. A cycle consists of a
 *       certain number of sample points set in `smpSz2`. The counting of cycles
 *       does not begin until the adaptive stage of every Markov chain enters
 *       the stage of a regular MCMC sampling. Setting MaxPSP to a low value may
 *       cause an incomplete search or unreliable information on the partition
 *       of parameter space. The default value is 6.
 *     - iniJmp: Initial size of jumping distribution with which the Markov
 *       chain in a newly found region starts, which is the radius of a uniform
 *       density over a hyper-sphere. `iniJump` does not depend on the scale of
 *       parameters, which is normalized, but is still problem-dependent and
 *       should affect the efficiency of PSP. The default value is 0.1.
 *     - smpSz1: Sizes of samples used to adapt the Markov chain of each
 *       discovered region. The adaptation is performed in two stages each of
 *       which consists of a few sampling cycles. `smpSz1` defines the size of
 *       each cycle in the Level 1 stage. The default value is
 *       ceil(100*1.2^NDIM).
 *     - smpSz2: Defines the size of each cycle in the Level 2 stage. Must be
 *       larger than the former value. Also serves as the size of regular MCMC
 *       cycles following the adaptation (termination condition). The default
 *       value is ceil(200*1.2^NDIM).
 *     - vsmpsz: Size of sample used to estimate a region's volume. The default
 *       value is ceil(500*1.2^NDIM).
 *     - accurateVolEst: Whether or not to perform an additional hit-or-miss
 *       Monte Carlo integration after the search process to estimate the region
 *       volume, which results in a better estimate.
 */
int PSP_Get_Regions(PSP_Handle handle,
                    PSP_Sampling_Callback sampling_callback,
                    int num_start_points,
                    Fixed *start_points,
                    Fixed *min_coords,
                    Fixed *max_coords,
                    PSP_Options options,
                    PSP_Result_Mode result_mode);

/**
 * Builds a partition of the space according to the sampled regions. Must be
 * called only after using `PSP_Get_Regions`.
 *
 * This method creates a binary space partitioning with SVM to estimate the
 * plane of separation between half-spaces.
 */
int PSP_Build_Partition_KdSVM(PSP_Handle handle,
                              PSP_KdSVMTree* tree);

#ifdef __cplusplus
}
#endif


#endif

/* EOF */
