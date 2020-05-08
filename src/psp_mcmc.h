#ifndef PSP_MCMC_H
#define PSP_MCMC_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
#include <vector>
#include <functional>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_MALLOC_ALREADY_ALIGNED 0
#define EIGEN_DONT_VECTORIZE
#define EIGEN_MAX_ALIGN_BYTES 0
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>

using Point = Eigen::VectorXd;
using Points = std::vector<Point>;
using Pattern = size_t;
using Model = std::function<Pattern(Point)>;


namespace PSP {
    struct too_many_patterns : public std::exception {
        using std::exception::exception;
    };
};


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
    unsigned int maxPatterns;
} PSP_Options;

typedef enum PSP_Result_Mode_ {
    PSP_RESULT_OVERWRITE,
    PSP_RESULT_APPEND,
    PSP_RESULT_COMBINE  //XXX: Only correctly combines the lists of sampled points!
} PSP_Result_Mode;

#ifdef __cplusplus
}

struct PSP_Result {
    std::vector<Pattern> patterns;
    std::vector<Points> xs;
    std::vector<Eigen::VectorXd> xMean;
    std::vector<Eigen::MatrixXd> xCovMat;
};

size_t nDim(PSP_Result const& psp_result);

PSP_Result psp_mcmc(Model model, Eigen::MatrixXd x0, Eigen::MatrixX2d xBounds, PSP_Options options = PSP_Options());
#endif

#endif

/* EOF */
