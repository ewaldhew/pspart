#include <iostream>
#include "pspart.h"


size_t sampl(void* sc, long* pnt)
{
    int d = 4;
    Point x = Eigen::Map<Eigen::VectorX<Fixed>>(pnt, 4).cast<double>() / 65536.0;
    if (x.cwiseAbs().sum() < 1) {
        return pow(2,d);
    } else {
        auto bits = Eigen::VectorXi((x.array() < 0).cast<int>());
        int dec = 0;
        for (int i = 0; i < bits.size(); i++) {
            dec |= bits.data()[i] << i;
        }
        return dec;
    }
}

int main()
{
    PSP_Handle hn = PSP_New(4);
    PSP_Sampling_CallbackRec cb{hn, sampl};
    Fixed x0[8] = { 12,0,0,0, 12,65536,65536,65536 };
    Fixed xm[4] = { 12,-65536,-65536,-65536 };
    Fixed xM[4] = { 12,65536,65536,65536 };
    PSP_Get_Regions(hn, &cb, 2, x0, xm, xM, {}, PSP_RESULT_OVERWRITE);
    PSP_Get_Regions(hn, &cb, 2, x0, xm, xM, {}, PSP_RESULT_APPEND);
    PSP_Get_Regions(hn, &cb, 2, x0, xm, xM, {}, PSP_RESULT_COMBINE);
    PSP_Close(hn);
}
