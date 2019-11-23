#ifndef BUILDPART_COMMON_H
#define BUILDPART_COMMON_H

#include "svm.h"

#ifdef __cplusplus
#include <iostream>

struct svm_model* train_svm(const struct svm_problem* problem, struct svm_parameter& param);

#endif

#endif

/* EOF */
