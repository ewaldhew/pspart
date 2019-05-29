#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <random>

#define PI 3.14159265358979323846264338327950288

#include <time.h>       /* time */
#define TIME_NOW time(NULL)

#include "psp_mcmc.h"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;


struct MarkovChain {
    MarkovChain(int sampleCount = 0, double optJump = 0, int level = 0, int alp = 0)
    :
    sampleCount(sampleCount),
    optJump(optJump),
    level(level),
    alp(alp) {};

    int sampleCount;
    double optJump;
    int level;
    int alp;
};

struct Region {
    Region(Point x,
           Pattern pattern)
    :
    xs(Points{x}), pattern(pattern), mc({})
    {
        int nDim = x.size();
        xsum = VectorXd::Zero(nDim);
        xcsum = MatrixXd::Zero(nDim, nDim);
    };

    Points xs;
    Pattern pattern;
    VectorXd xsum;
    MatrixXd xcsum;
    MarkovChain mc;

    friend struct Regions;

private:
    Region(Points xs,
           Pattern pattern,
           VectorXd xsum,
           MatrixXd xcsum,
           MarkovChain mc)
    :
    xs(xs), pattern(pattern), xsum(xsum), xcsum(xcsum), mc(mc)
    { };
};

struct Regions {
    std::vector<Points> xs;
    std::vector<Pattern> patterns;
    std::vector<VectorXd> xsum;
    std::vector<MatrixXd> xcsum;
    std::vector<int> sampleCount;
    std::vector<double> optJump;
    std::vector<int> levels;
    std::vector<int> alps;

    void push_back(Region new_region)
    {
        xs.push_back(new_region.xs);
        patterns.push_back(new_region.pattern);
        xsum.push_back(new_region.xsum);
        xcsum.push_back(new_region.xcsum);
        sampleCount.push_back(new_region.mc.sampleCount);
        optJump.push_back(new_region.mc.optJump);
        levels.push_back(new_region.mc.level);
        alps.push_back(new_region.mc.alp);
    }

    int size()
    {
        return xs.size();
    }

    Region operator[](int i)
    {
        return { xs[i], patterns[i], xsum[i], xcsum[i], { sampleCount[i], optJump[i], levels[i], alps[i] } };
    }
};

PSP_Result psp_mcmc(Model model, MatrixXd x0, MatrixX2d xBounds, PSP_Options options)
{
    std::default_random_engine generator(TIME_NOW);
    std::normal_distribution<double> randn;
    std::uniform_real_distribution<double> rand;

    Point xMin = xBounds.col(0);
    Point xMax = xBounds.col(1);
    VectorXd xRange = xMax - xMin;
    if (x0.rows() != xBounds.rows()) {
        throw std::invalid_argument("Dimension mismatch.");
    }
    if ((xRange.array() <= 0).any()) {
        throw std::invalid_argument("Invalid bounds.");
    }
    if (x0.size() == 0 || ( x0.array() < xMin.replicate(1, x0.cols()).array()
                         || x0.array() > xMax.replicate(1, x0.cols()).array() ).any()) {
        throw std::invalid_argument("Invalid starting point.");
    }

    int nDim = xBounds.rows();

    /* Default values of options */
    int maxPsp = options.maxPsp < 0 ? 6 : options.maxPsp;
    double iniJmp = options.iniJmp < 0 ? .1 : options.iniJmp;
    int smpSz1 = options.smpSz1 < 0 ? ceil(100 * pow(1.2, nDim)) : options.smpSz1;
    int smpSz2 = options.smpSz2 < 0 ? ceil(200 * pow(1.2, nDim)) : options.smpSz2;
    int vsmpsz = options.vsmpsz < 0 ? ceil(500 * pow(1.2, nDim)) : options.vsmpsz;

    /* MCMC-based Parameter Space Partitioning Algorithm */

    std::unordered_set<Pattern> foundPatterns;

    Regions regions;
    std::vector<std::pair<time_t, int>> searchTime;

    time_t t0 = TIME_NOW;
    int numTrials = 0;

    std::cout << "=================================================================\n"
                 "PSP SEARCH STARTS...\n\n";

    for (int i = 0; i < x0.cols(); i++) {
        Point y = x0.col(i);
        Pattern currPtn = model(y);

        if (foundPatterns.insert(currPtn).second) {
            regions.push_back({ y, currPtn });
            searchTime.push_back({ TIME_NOW - t0, numTrials });

            std::cout << "New data pattern found: " << currPtn << "\n";
            std::cout << "w/ supplied starting point(s), Total elapsed time: " <<
                searchTime.back().first << " secs (" << numTrials << " trials)\n";
        }
    }

    int iterCount1 = 0;
    int iterCount2 = 0;
    int cnt1 = TIME_NOW;
    int cnt2 = TIME_NOW;

    int maxpspp = maxPsp * smpSz2;
    int minLevel = 0;

    while (minLevel < 2 ||
           *std::min_element(regions.sampleCount.begin(),
                             regions.sampleCount.end()) <= maxpspp) {
        int regionIdx = 0;
        for (int i = 0; i < regions.size(); i++) {
            if (regions.levels[i] == minLevel &&
                regions.sampleCount[i] < regions.sampleCount[regionIdx]) {
                regionIdx = i;
            }
        }
        regions.sampleCount[regionIdx]++;

        VectorXd rnd1 = VectorXd::NullaryExpr(nDim, [&]() { return randn(generator); });
        VectorXd rnd2 = pow(rand(generator), 1 / nDim) * rnd1.normalized();
        VectorXd jump = xRange.cwiseProduct(iniJmp * pow(2, regions.optJump[regionIdx]) * rnd2);
        Point y = regions.xs[regionIdx].back() + jump;
        numTrials++;

        if ((xMin.array() <= y.array()).all() && (y.array() <= xMax.array()).all()) {
            Pattern currPtn = model(y);

            if ((currPtn == regions.patterns[regionIdx])) {
                regions.xs[regionIdx].push_back(y);
                regions.alps[regionIdx]++;
            } else {
                if (foundPatterns.insert(currPtn).second) {
                    regions.push_back({ y, currPtn });
                    searchTime.push_back({ TIME_NOW - t0, numTrials });

                    iterCount1 = iterCount2 = 0;
                    cnt1 = cnt2 = TIME_NOW;

                    std::cout << "New data pattern found: " << currPtn << "\n";
                    std::cout << "PSP, Total elapsed time: " <<
                        searchTime.back().first << " secs (" << numTrials << " trials)\n";
                }
            }
        }

        switch (regions.levels[regionIdx]) {
        case 0:
        {
            double tmp = regions.sampleCount[regionIdx] / (double)smpSz1;

            if (tmp == ceil(tmp)) {
                double acrate = regions.alps[regionIdx] / (double)smpSz1;
                regions.alps[regionIdx] = 0;

                std::cout << "\nLevel 1 adaptation of MCMC in Region #" << regionIdx << '\n'
                          << "Cycle #" << tmp << ", Acceptance rate: " << acrate << '\n';

                if (acrate < .12)
                {
                    if (regions.optJump[regionIdx] > 0) {
                        regions.optJump[regionIdx] -= .5;
                        regions.levels[regionIdx] = 1;
                        regions.sampleCount[regionIdx] = 0;
                    } else {
                        regions.optJump[regionIdx] -= 1;
                    }
                } else if (acrate >= .12 && acrate < .36) {
                    regions.levels[regionIdx] = 1;
                    regions.sampleCount[regionIdx] = 0;
                } else if (acrate >= .36) {
                    if (regions.optJump[regionIdx] < 0) {
                        regions.optJump[regionIdx] += .5;
                        regions.levels[regionIdx] = 1;
                        regions.sampleCount[regionIdx] = 0;
                    } else {
                        regions.optJump[regionIdx] += 1;
                    }
                }
            }
        } break;

        case 1:
        {
            double tmp = regions.sampleCount[regionIdx] / (double)smpSz2;

            if (tmp == ceil(tmp)) {
                double acrate = regions.alps[regionIdx] / (double)smpSz2;
                regions.alps[regionIdx] = 0;

                std::cout << "\nLevel 2 adaptation of MCMC in Region #" << regionIdx << '\n'
                          << "Cycle #" << tmp << ", Acceptance rate: " << acrate << '\n';

                if (acrate < .15) {
                    regions.optJump[regionIdx] = regions.optJump[regionIdx] - .25 / ceil(tmp / 2);
                    if (tmp == 4) {
                        regions.levels[regionIdx] = 2;
                        regions.sampleCount[regionIdx] = 0;
                    }
                } else if (acrate >= .15 && acrate < .19) {
                    regions.optJump[regionIdx] -= .125;
                    regions.levels[regionIdx] = 2;
                    regions.sampleCount[regionIdx] = 0;
                } else if (acrate >= .19 && acrate < .24) {
                    regions.levels[regionIdx] = 2;
                    regions.sampleCount[regionIdx] = 0;
                } else if (acrate >= .24 && acrate < .3) {
                    regions.optJump[regionIdx] += .125;
                    regions.levels[regionIdx] = 2;
                    regions.sampleCount[regionIdx] = 0;
                } else if (acrate >= .3) {
                    regions.optJump[regionIdx] = regions.optJump[regionIdx] + .25 / ceil(tmp / 2);
                    if (tmp == 4) {
                        regions.levels[regionIdx] = 2;
                        regions.sampleCount[regionIdx] = 0;
                    }
                }
            }
        } break;

        case 2:
        {
            double tmp = regions.sampleCount[regionIdx] / (double)smpSz2;

            if (regions.sampleCount[regionIdx] == 1) {
                std::cout << "Adaptation of MCMC in Region #" << regionIdx << " finished.\n";
            } else if (tmp == ceil(tmp)) {
                double acrate = regions.alps[regionIdx] / (double)regions.sampleCount[regionIdx];
                std::cout << "\nMonitoring after adaptation in Region #" << regionIdx << '\n'
                          << "Cycle #" << tmp << ", Acceptance rate (cumulative): " << acrate << '\n';
            }

            Point lastPoint = regions.xs[regionIdx].back();
            regions.xsum[regionIdx] += lastPoint;
            regions.xcsum[regionIdx] += lastPoint * lastPoint.transpose();
        } break;
        }

        iterCount1++;
        minLevel = *std::min_element(regions.levels.begin(), regions.levels.end());

        {
            auto minmaxSmpCnt = std::minmax_element(regions.sampleCount.begin(),
                                                    regions.sampleCount.end());

            if (minLevel < 2 || *minmaxSmpCnt.second - *minmaxSmpCnt.first > 1) {
                iterCount2 = 0;
                cnt2 = TIME_NOW;
            } else {
                iterCount2++;
            }
        }

    }

    std::vector<Pattern> resultPatterns(regions.patterns);
    std::vector<Points> resultXs(regions.xs);
    std::vector<VectorXd> resultXMean;
    std::vector<MatrixXd> resultXCovMat;
    resultXMean.reserve(regions.size());
    resultXCovMat.reserve(regions.size());

    for (int i = 0; i < regions.size(); i++) {
        double smpCnt = (double)regions.sampleCount[i];
        VectorXd xsum = regions.xsum[i];
        resultXMean.push_back(xsum / smpCnt);
        resultXCovMat.push_back(regions.xcsum[i] / smpCnt
                                - (xsum * xsum.transpose()) / smpCnt*smpCnt);
    }

    std::vector<double> logvol(regions.size(), 0);
    double nHalf = nDim * 0.5;
    double nFloor = floor(nHalf);
    double offset = nHalf == nFloor
                    ? nHalf * log(PI) - lgamma(nHalf + 1)
                    : nDim * log(2) + lgamma(nFloor + 1) - lgamma(nDim + 1) + nFloor * log(PI);
    for (int i = 0; i < regions.size(); i++) {
        logvol[i] = offset + .5 * (log(nDim + 2)
            + resultXCovMat[i].eigenvalues().array().log())
            .sum().real();
    }

    if (options.accurateVolEst) {
        std::cout << "\nVolume estimation by hit-or-miss method begins...\n";

        for (int i = 0; i < regions.size(); i++) {
            int nHit = 0;

            std::cout << "Estimating the volume of Region #" << i << std::endl;

            MatrixXd sqrtm = ((nDim + 2) * resultXCovMat[i]).sqrt();
            for (int j = 0; j < vsmpsz; j++) {
                VectorXd rnd1 = VectorXd::NullaryExpr(nDim, [&]() { return randn(generator); });
                VectorXd rnd2 = pow(rand(generator), 1 / nDim) * rnd1.normalized();
                Point y = resultXMean[i] + sqrtm * rnd2;

                if ((xMin.array() <= y.array()).all() && (y.array() <= xMax.array()).all()) {
                    Pattern cPtn = model(y);
                    if (cPtn == regions.patterns[i]) {
                        nHit++;
                    }
                }
            }

            logvol[i] += log(nHit) - log(vsmpsz);
        }

        std::cout << "...Volume estimation terminated for all regions.\n";
    }

    searchTime.push_back({ TIME_NOW - t0, numTrials });
    std::cout << "\nPSP SEARCH TERMINATED.\n" \
                 "TOTAL " << regions.size() << " DATA PATTERNS FOUND.\n" \
                 "TOTAL " << searchTime.back().first << " secs ("
                 << numTrials << " trials) ELASPED.\n" \
                 "=================================================================\n";

    return { std::move(resultPatterns), std::move(resultXs),
             std::move(resultXMean), std::move(resultXCovMat) };
}
