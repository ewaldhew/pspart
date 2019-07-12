#include <stdio.h>
#include <stdlib.h>
#include <pspart.h>


#define DIM 3
#define KD 0

size_t sampl(void* sc, long* pnt)
{
    long sum = 0;
    size_t dec = 0;
    for (int i = 0; i < DIM; i++) {
        sum += abs(pnt[i]);
        dec |= (pnt[i] < 0 ? 0L : 1L)  << i;
    }
    return 100 + (sum < 65536 ? 16 : dec);
}

int main()
{
    PSP_Handle hn = PSP_New(DIM);
    PSP_Sampling_CallbackRec cb = {hn, sampl};
    Fixed x0[DIM] = { 0,0,0 };
    Fixed xm[DIM] = { -65536,-65536,-65536 };
    Fixed xM[DIM] = { 65536,65536,65536 };
    PSP_Options options = {0};
    PSP_Get_Regions(hn, &cb, 1, x0, xm, xM, options, PSP_RESULT_APPEND);
    struct svm_parameter params = {.svm_type=C_SVC, .kernel_type=POLY, .degree=2, .gamma=1.0/DIM, .C=10000,
      .cache_size=1000, .eps=1e-3};
    PSP_Configure_SVM(hn, &params);
#if KD
    PSP_KdSVMTree tree = NULL;
    PSP_Build_Partition_KdSVM(hn, &tree);
#else
    PSP_MCSVM svm = NULL;
    PSP_Build_Partition_MCSVM(hn, &svm);
#endif

    psp_dump_points(hn);
    fprintf(stdout, "Enter coordinates to test:\n"); fflush(stdout);

    while (1) {
        long x1[DIM];
        double xd1[DIM];
        for (int i = 0; i < DIM; i++) {
            int a;
            if (fscanf(stdin, "%d", &a) == EOF) exit(0);
            fprintf(stdout, "%d ", a);
            x1[i] = a;
            xd1[i] = a / 65536.0;
        }
        struct svm_node node = { DIM, xd1 };
#if KD
        PSP_KdSVMTree curr = tree;
        while (curr->node.left && curr->node.right) {
            if (svm_predict(curr->data.model, &node) > 0) {
                curr = (PSP_KdSVMTree)curr->node.left;
            } else {
                curr = (PSP_KdSVMTree)curr->node.right;
            }
        }
        size_t predicted = curr->data.pattern;
#else
        PSP_MCSVM curr = svm;
        size_t predicted = svm_predict(curr->model, &node);
#endif
        size_t actual = sampl(NULL, x1);
        fprintf(stdout, "%ld -> %ld\n", actual, predicted);
    }
}
