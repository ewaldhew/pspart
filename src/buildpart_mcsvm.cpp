#include <exception>

#include "debug.h"
#include "buildpart_mcsvm.h"


struct MCSVM_Internal : Node_Internal {
    using Node_Internal::Node_Internal;
    PSP_MCSVM transformed;
    svm_model* model;
    svm_problem problem;

    ~MCSVM_Internal();
};
using MCSVM_InternalPtr = std::shared_ptr<MCSVM_Internal>;

MCSVM_Internal::~MCSVM_Internal()
{
    delete transformed;
    svm_destroy_param(&model->param);
    svm_free_and_destroy_model(&model);
    delete[] problem.y;
    for (int i = 0; i < problem.l; i++) {
        delete[] problem.x[i].values;
    }
    delete[] problem.x;
}

static inline
struct svm_model* build_svm(PSP_Result const& regions,
                            svm_parameter const* parameters,
                            svm_problem* problem)
{
    size_t dim = nDim(regions);

    svm_parameter param = {};
    if (!parameters) {
        param.svm_type = NU_SVC;
        param.kernel_type = POLY;
        param.degree = 3;
        param.gamma = 100;
        param.cache_size = 100;
        param.nu = 1e-6;
        param.eps = 1e-3;
        param.shrinking = 1;
    } else {
        param = *parameters;
    }

    int num_points = 0;
    for (size_t i = 0; i < regions.patterns.size(); i++) {
        num_points += regions.xs[i].size();
    }

    problem->l = num_points;
    problem->x = new struct svm_node[num_points];
    problem->y = new double[num_points];

    int i = 0;
    for (size_t j = 0; j < regions.patterns.size(); j++) {
        for (auto & point : regions.xs[j]) {
            assert(i < num_points);

            svm_node node;
            node.dim = dim;
            node.values = new double[dim];
            Eigen::Map<Point>(node.values, dim) = point;

            problem->x[i] = node;
            problem->y[i] = regions.patterns[j];
            i++;
        }
    }

    const char* error_msg = svm_check_parameter(problem, &param);
    if (error_msg)
        throw std::invalid_argument(error_msg);

    return train_svm(problem, param);
}

static inline
MCSVM_InternalPtr build_mcsvm_internal(PSP_Result const& regions,
                                       svm_parameter const* param)
{
    svm_problem problem = {};

    MCSVM_InternalPtr result = std::make_shared<MCSVM_Internal>(MCSVM_InternalPtr(),
                                                                MCSVM_InternalPtr());
    result->model = build_svm(regions, param, &problem);
    result->problem = problem;

    return result;
}

static inline
PSP_MCSVM transform_mcsvm(Node_InternalPtr const& mcsvm)
{
    if (mcsvm == nullptr) {
        return NULL;
    }

    MCSVM_InternalPtr node = std::static_pointer_cast<MCSVM_Internal>(mcsvm);

    PSP_MCSVM result = new PSP_MCSVMRec{};
    result->model = node->model;
    node->transformed = result;
    return result;
}

PSP_MCSVM build_mcsvm(PSP_Result data,
                      svm_parameter const* param,
                      PSP_Memory memory)
{
    memory->mcsvm = build_mcsvm_internal(data, param);

    return transform_mcsvm(memory->mcsvm);
}
