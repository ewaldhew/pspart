#include <exception>
#include <numeric>

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
    svm_destroy_param(&data.model->param);
    svm_free_and_destroy_model(&data.model);
    delete[] problem.y;
    for (int i = 0; i < problem.l; i++) {
        delete[] problem.x[i].values;
    }
    delete[] problem.x;
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
struct svm_model* build_svm(PSP_Result const& regions,
                            std::vector<size_t>::const_iterator begin,
                            std::vector<size_t>::const_iterator mid,
                            std::vector<size_t>::const_iterator end,
                            svm_problem* problem)
{
    size_t dim = nDim(regions);

    svm_parameter param = {};
    param.kernel_type = POLY;
    param.degree = 2;
    param.gamma = 1.0 / dim;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.shrinking = 1;

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
            problem->y[i] = it < mid ? -1 : 1;
            i++;
        }
    }

    const char* error_msg = svm_check_parameter(problem, &param);
    if (error_msg)
        throw std::runtime_error(error_msg);

    return svm_train(problem, &param);
}

static inline
KdSVM_InternalPtr build_kdsvm_internal(PSP_Result const& regions,
                                       std::vector<size_t>::iterator begin,
                                       std::vector<size_t>::iterator end,
                                       size_t dim = 0)
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
        // select dimension - simply cycle it here
        size_t dim = dim % nDim(regions);

        // find median of points in the chosen dimension
        auto mid = begin + (end - begin) / 2;
        std::nth_element(begin, mid, end, [regions, dim](size_t lhs, size_t rhs) {
            return regions.xMean[lhs][dim] < regions.xMean[rhs][dim];
        });

        // build the separating plane
        data.model = build_svm(regions, begin, mid, end, &problem);

        left = build_kdsvm_internal(regions, begin, mid, dim + 1);
        right = build_kdsvm_internal(regions, mid, end, dim + 1);
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
                          PSP_Memory memory)
{
    std::vector<size_t> indices(data.patterns.size());
    std::iota(std::begin(indices), std::end(indices), 0);

    memory->kdsvm = build_kdsvm_internal(data, std::begin(indices), std::end(indices));

    return transform_kdsvm(memory->kdsvm);
}
