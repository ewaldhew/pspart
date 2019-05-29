#ifndef PSP_MCMC_H
#define PSP_MCMC_H

#include <vector>
#include <functional>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>

using Point = Eigen::VectorXd;
using Points = std::vector<Point>;
using Pattern = size_t;
using Model = std::function<Pattern(Point)>;

#define PSP_OPTION_NOT_SET -1
struct PSP_Options {
    int maxPsp = PSP_OPTION_NOT_SET;
    double iniJmp = PSP_OPTION_NOT_SET;
    double smpSz1 = PSP_OPTION_NOT_SET;
    double smpSz2 = PSP_OPTION_NOT_SET;
    double vsmpsz = PSP_OPTION_NOT_SET;
    bool accurateVolEst = false;
};

struct PSP_Result {
    std::vector<Pattern> resultPatterns;
    std::vector<Points> resultXs;
    std::vector<Eigen::VectorXd> resultXMean;
    std::vector<Eigen::MatrixXd> resultXCovMat;
};


PSP_Result psp_mcmc(Model model, Eigen::MatrixXd x0, Eigen::MatrixX2d xBounds, PSP_Options options = PSP_Options());

#endif

/* EOF */
