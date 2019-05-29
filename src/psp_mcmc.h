#ifndef PSP_MCMC_H
#define PSP_MCMC_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
#include <vector>
#include <functional>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>

using Point = Eigen::VectorXd;
using Points = std::vector<Point>;
using Pattern = size_t;
using Model = std::function<Pattern(Point)>;


extern "C"
{
#endif

#define PSP_OPTION_NOT_SET -1
typedef struct PSP_Options_ {
    int maxPsp;
    double iniJmp;
    double smpSz1;
    double smpSz2;
    double vsmpsz;
    bool accurateVolEst;
} PSP_Options;

typedef enum PSP_Result_Mode_ {
    PSP_RESULT_OVERWRITE,
    PSP_RESULT_APPEND
} PSP_Result_Mode;

#ifdef __cplusplus
}

struct PSP_Result {
    std::vector<Pattern> patterns;
    std::vector<Points> xs;
    std::vector<Eigen::VectorXd> xMean;
    std::vector<Eigen::MatrixXd> xCovMat;
};


PSP_Result psp_mcmc(Model model, Eigen::MatrixXd x0, Eigen::MatrixX2d xBounds, PSP_Options options = PSP_Options());
#endif

#endif

/* EOF */
