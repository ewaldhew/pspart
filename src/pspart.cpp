#include "debug.h"
#include "pspart.h"


static int HandleExceptions() noexcept
{
    try { throw; }

    catch (std::bad_alloc const&)
    {
        return ENOMEM;
    }
    catch (std::invalid_argument const&)
    {
        return EINVAL;
    }
    catch (...)
    {
        return ERR_UNHANDLED_EXCEPTION;
    }
}


struct PSP_Handle_ {
    size_t n_dim;
    PSP_Result psp_regions;
    svm_parameter* svm_params;
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

        PSP_Result const& result = psp_mcmc(model, x0, xb, options);

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

        case PSP_RESULT_COMBINE:
        {
            std::vector<size_t> idxs(result.patterns.size());
            std::transform(result.patterns.begin(), result.patterns.end(),
                           idxs.begin(),
                           [handle](Pattern const& ptn) {
                               auto it = std::find(handle->psp_regions.patterns.rbegin(),
                                                   handle->psp_regions.patterns.rend(),
                                                   ptn);
                               return handle->psp_regions.patterns.rend() - it - 1;
                           });

            for (int i = 0; i < result.patterns.size(); i++) {
                int idx = idxs[i];

                if (idx == -1) {
                    handle->psp_regions.patterns.push_back(result.patterns[i]);
                    handle->psp_regions.xs.push_back(result.xs[i]);
                    handle->psp_regions.xMean.push_back(result.xMean[i]);
                    handle->psp_regions.xCovMat.push_back(result.xCovMat[i]);
                } else {
                    int a = handle->psp_regions.xs[idx].size();
                    int b = result.xs[i].size();
                    Point x = handle->psp_regions.xMean[idx];
                    Point y = result.xMean[i];

                    append(handle->psp_regions.xs[idx], result.xs[i]);
                    handle->psp_regions.xMean[idx] = (a*x + b*y) / (a + b);
                }
            }
        }
            break;
        }
    } catch (...) {
        return HandleExceptions();
    }

    return 0;
}


extern "C"
int PSP_Configure_SVM(PSP_Handle handle,
                      struct svm_parameter* params)
{
    if (!handle)
        return EINVAL;

    handle->svm_params = params;

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
        *tree = build_kdsvm(handle->psp_regions, handle->svm_params, handle->memory);
    } catch (...) {
        return HandleExceptions();
    }

    return 0;
}

extern "C"
int PSP_Build_Partition_MCSVM(PSP_Handle handle,
                              PSP_MCSVM* node)
{
    if (!handle || !node)
        return EINVAL;

    try {
        if (!handle->memory)
            handle->memory = new PSP_MemoryRec{};
        *node = build_mcsvm(handle->psp_regions, handle->svm_params, handle->memory);
    } catch (...) {
        return HandleExceptions();
    }

    return 0;
}


extern "C"
void psp_dump_points(PSP_Handle handle)
{
#ifdef DEBUG
    std::cout << "Sampled points dump:\n";
    for (size_t i = 0; i < handle->psp_regions.patterns.size(); i++) {
        std::cout << handle->psp_regions.patterns[i] << ' '
                  << handle->psp_regions.xs[i].size() << '\n'
                  << unmap_coord(handle->psp_regions.xMean[i]).transpose() << '\n';
        for (auto & x : handle->psp_regions.xs[i]) {
            std::cout << unmap_coord(x).transpose() << '\n';
        }
    }
#endif
}


/* EOF */
