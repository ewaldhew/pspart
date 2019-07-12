#include <exception>
#include <numeric>

#include "debug.h"
#include "buildpart_kdsvm.h"


struct KdSVM_Internal : Node_Internal {
    using Node_Internal::Node_Internal;
    PSP_KdSVMTree_Data data;
    svm_problem problem;

    ~KdSVM_Internal();
};
using KdSVM_InternalPtr = std::shared_ptr<KdSVM_Internal>;

KdSVM_Internal::~KdSVM_Internal()
{
    if (left != nullptr || right != nullptr) {
        svm_destroy_param(&data.model->param);
        svm_free_and_destroy_model(&data.model);
        delete[] problem.y;
        for (int i = 0; i < problem.l; i++)
        {
            delete[] problem.x[i].values;
        }
        delete[] problem.x;
    }
}

static inline
KdSVM_InternalPtr KdSVM_InternalPtr_Make()
{
    return KdSVM_InternalPtr();
}

static inline
KdSVM_InternalPtr KdSVM_InternalPtr_Make(KdSVM_InternalPtr left,
                                         KdSVM_InternalPtr right)
{
    return std::make_shared<KdSVM_Internal>(left, right);
}

static inline
bool check_model(svm_model* model,
                 double coef_max)
{
    if (!model)
        return false;

    if (coef_max <= 0)
        return true;

    if (model->rho[0] < -coef_max || model->rho[0] > coef_max)
        return false;

    for (int j = 0; j < model->l; j++)
        if (model->sv_coef[0][j] < -coef_max || model->sv_coef[0][j] > coef_max)
            return false;

    return true;
}

static inline
struct svm_model* build_svm(PSP_Result const& regions,
                            std::vector<size_t>::const_iterator begin,
                            std::vector<size_t>::const_iterator mid,
                            std::vector<size_t>::const_iterator end,
                            svm_parameter const* parameters,
                            svm_problem* problem)
{
    DEBUG_LOG("SVM: { ");
    for (auto it = begin; it < mid; it++) {
        DEBUG_LOG(regions.patterns[*it] << " ");
    }
    DEBUG_LOG("} vs. { ");
    for (auto it = mid; it < end; it++) {
        DEBUG_LOG(regions.patterns[*it] << " ");
    }
    DEBUG_LOG("}\n");

    svm_model* model = NULL;
    int num_retries = 0;

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
    for (auto it = begin; it < end; it++) {
        num_points += regions.xs[*it].size();
    }

    problem->l = num_points;
    problem->x = new struct svm_node[num_points];
    problem->y = new double[num_points];

    int i = 0;
    for (auto it = begin; it < end; it++) {
        for (auto & point : regions.xs[*it]) {
            assert(i < num_points);

            svm_node node;
            node.dim = dim;
            node.values = new double[dim];
            Eigen::Map<Point>(node.values, dim) = point;

            problem->x[i] = node;
            problem->y[i] = it < mid ? 1 : -1;
            i++;
        }
    }

    const char* error_msg = svm_check_parameter(problem, &param);
    if (error_msg)
        throw std::runtime_error(error_msg);

    do {
        if (model) {
            fprintf(stderr, "build_svm: Coefficients too large, retrying...\n");

            if (num_retries == 0) {
                param.svm_type = C_SVC;
                param.C = param.coef_max;
            } else {
                param.C = param.C / 10;
            }
        }

        model = svm_train(problem, &param);

        if (num_retries > param.max_retries) {
            fprintf(stderr, "build_svm: Max retry count reached, giving up..\n");
            break;
        }
    } while (!check_model(model, param.coef_max));

    return model;
}

static inline
KdSVM_InternalPtr build_kdsvm_internal(PSP_Result const& regions,
                                       std::vector<size_t>::iterator begin,
                                       std::vector<size_t>::iterator end,
                                       svm_parameter const* param,
                                       size_t dim_ = 0)
{
    PSP_KdSVMTree_Data data;
    KdSVM_InternalPtr left, right;
    svm_problem problem = {};

    if (begin >= end) {
        return KdSVM_InternalPtr_Make();
    }

    if (begin == end - 1) {
        data.pattern = regions.patterns[*begin];

        left = KdSVM_InternalPtr_Make();
        right = KdSVM_InternalPtr_Make();
    } else {
#if 0
        // select dimension - simply cycle it here
        size_t dim = dim_ % nDim(regions);
#else
        // select dimension - find the longest bounding box edge
        size_t dim;
        double max_range = 0;

        for (int i = 0; i < nDim(regions); i++) {
            auto minmax = std::minmax_element(regions.xMean.begin(),
                                              regions.xMean.end(),
                                              [i](Point const &lhs, Point const &rhs)
                                              { return lhs[i] < rhs[i]; });

            double range = (*minmax.second)[i] - (*minmax.first)[i];
            if (max_range < range) {
                max_range = range;
                dim = i;
            }
        }
#endif

        // find median of points in the chosen dimension
        auto mid = begin + (end - begin) / 2;
        std::nth_element(begin, mid, end, [regions, dim](size_t lhs, size_t rhs) {
            return regions.xMean[lhs][dim] < regions.xMean[rhs][dim];
        });

        // build the separating plane
        data.model = build_svm(regions, begin, mid, end, param, &problem);

        left = build_kdsvm_internal(regions, begin, mid, param, dim + 1);
        right = build_kdsvm_internal(regions, mid, end, param, dim + 1);
    }

    KdSVM_InternalPtr result = KdSVM_InternalPtr_Make(left, right);
    result->data = data;
    result->problem = problem;

    return result;
}

static inline
PSP_KdSVMTree transform_kdsvm(Node_InternalPtr const& kdsvm)
{
    if (kdsvm == nullptr) {
        return NULL;
    }

    KdSVM_InternalPtr tree = std::static_pointer_cast<KdSVM_Internal>(kdsvm);

    PSP_KdSVMTree result = new PSP_KdSVMTreeRec;
    result->data = tree->data;
    result->node.left = (PSP_Node)transform_kdsvm(tree->left);
    result->node.right = (PSP_Node)transform_kdsvm(tree->right);
    return result;
}

PSP_KdSVMTree build_kdsvm(PSP_Result data,
                          svm_parameter const* param,
                          PSP_Memory memory)
{
    std::vector<size_t> indices(data.patterns.size());
    std::iota(std::begin(indices), std::end(indices), 0);

    memory->kdsvm = build_kdsvm_internal(data, std::begin(indices), std::end(indices), param);

    return transform_kdsvm(memory->kdsvm);
}
