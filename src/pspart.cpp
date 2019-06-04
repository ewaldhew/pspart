#include "pspart.h"


static int HandleExceptions() noexcept
{
    try { throw; }

    catch (const std::bad_alloc &)
    {
        return ENOMEM;
    }
    catch (...)
    {
        return ERR_UNHANDLED_EXCEPTION;
    }
}


struct PSP_Handle_ {
    size_t n_dim;
    PSP_Result psp_regions;
    PSP_Memory memory;
};

using Point_Fixed = Eigen::VectorX<Fixed>;

static inline
Point map_coord(PSP_Handle handle, Fixed* coord)
{
    return Eigen::Map<Point_Fixed>(coord, handle->n_dim).cast<double>() / 65536.0;
}

static inline
Point_Fixed unmap_coord(Point const& coord)
{
    return (coord * 65536).cast<Fixed>();
}

template <typename T>
static inline
void append(std::vector<T> & dest, std::vector<T> src)
{
    if (dest.empty()) {
        dest = std::move(src);
    } else {
        dest.insert(std::end(dest),
                    std::make_move_iterator(std::begin(src)),
                    std::make_move_iterator(std::end(src)));
    }
}


extern "C"
PSP_Handle PSP_New(size_t dim)
{
    PSP_Handle handle = NULL;

    if (dim < 1)
        return NULL;

    try {
        handle = new PSP_Handle_{ dim };
    } catch (...) {
        fprintf(stderr, "PSP_New: failed with code %d",
                HandleExceptions());
        handle = NULL;
    }

    return handle;
}

extern "C"
void PSP_Close(PSP_Handle handle)
{
    delete handle->memory;
    delete handle;
}

extern "C"
int PSP_Get_Regions(PSP_Handle handle,
                    PSP_Sampling_Callback sampling_callback,
                    int num_start_points,
                    Fixed *start_points,
                    Fixed *min_coords,
                    Fixed *max_coords,
                    PSP_Options options,
                    PSP_Result_Mode result_mode)
{
    if (!handle || !sampling_callback->sampler)
        return EINVAL;

    try {
        auto model = [sampling_callback](Point x) {
            return sampling_callback->sampler(sampling_callback->sampling_context,
                                              unmap_coord(x).data());
        };

        Eigen::MatrixXd x0(handle->n_dim, num_start_points);
        for (int i = 0; i < num_start_points; i++) {
            x0.col(i) = map_coord(handle, start_points + i * handle->n_dim);
        }
        Eigen::MatrixX2d xb(handle->n_dim, 2);
        xb << map_coord(handle, min_coords), map_coord(handle, max_coords);

        PSP_Result result = psp_mcmc(model, x0, xb, options);

        switch (result_mode) {
        default:
        case PSP_RESULT_OVERWRITE:
            handle->psp_regions = result;
            break;

        case PSP_RESULT_APPEND:
            append(handle->psp_regions.patterns, result.patterns);
            append(handle->psp_regions.xs, result.xs);
            append(handle->psp_regions.xMean, result.xMean);
            append(handle->psp_regions.xCovMat, result.xCovMat);
            break;
        }
    } catch (...) {
        return HandleExceptions();
    }

    return 0;
}

extern "C"
int PSP_Build_Partition_KdSVM(PSP_Handle handle,
                              PSP_KdSVMTree* tree)
{
    if (!handle || !tree)
        return EINVAL;

    try {
        if (!handle->memory)
            handle->memory = new PSP_MemoryRec{};
        *tree = build_kdsvm(handle->psp_regions, handle->memory);
    } catch (...) {
        return HandleExceptions();
    }

    return 0;
}


/* EOF */
