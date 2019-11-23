#include "buildpart_common.h"

static inline
bool check_model(svm_model* model,
                 double coef_max)
{
    int num_classes, num_vectors, i, j;

    if (!model)
        return false;

    if (coef_max <= 0)
        return true;

    num_classes = model->nr_class;
    num_vectors = model->l;

    for (i = 0; i < (num_classes * (num_classes - 1)) / 2; i++)
        if (model->rho[i] < -coef_max || model->rho[i] > coef_max)
            return false;

    for (i = 0; i < num_classes - 1; i++)
        for (j = 0; j < num_vectors; j++)
            if (model->sv_coef[i][j] < -coef_max || model->sv_coef[i][j] > coef_max)
                return false;

    return true;
}

struct svm_model* train_svm(const struct svm_problem* problem,
                            struct svm_parameter& param)
{
    svm_model* model = NULL;
    int num_retries = 0;

    do {
        if (model) {
            fprintf(stderr, "build_svm: Coefficients too large, retrying...\n");

            if (num_retries == 0) {
                param.svm_type = NU_SVC;
            }

            param.nu = model->param.nu * 2;
            num_retries++;
        }

        if (num_retries > param.max_retries || param.nu >= 1.0) {
            param.svm_type = C_SVC;
            param.C = param.coef_max;
            model = svm_train(problem, &param);
            if (!check_model(model, param.coef_max)) {
                fprintf(stderr, "build_svm: Max retry count reached, giving up..\n");
                break;
            }
        }

        model = svm_train(problem, &param);

    } while (!check_model(model, param.coef_max));

    return model;
}
