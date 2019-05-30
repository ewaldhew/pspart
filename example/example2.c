#include <stdio.h>
#include <pspart.h>


#define DIM 2

size_t sampl(void* sc, long* pnt)
{
    size_t dec = 0;
    for (int i = 0; i < DIM; i++) {
        dec |= (pnt[i] < 0 ? 0L : 1L)  << i;
    }
    return dec;
}

int main()
{
    PSP_Handle hn = PSP_New(DIM);
    PSP_Sampling_CallbackRec cb = {hn, sampl};
    Fixed x0[DIM] = { 0,0 };
    Fixed xm[DIM] = { -65536,-65536 };
    Fixed xM[DIM] = { 65536,65536 };
    PSP_Options options = {0};
    PSP_Get_Regions(hn, &cb, 1, x0, xm, xM, options, PSP_RESULT_APPEND);
    PSP_KdSVMTree tree = NULL;
    PSP_Build_Partition_KdSVM(hn, &tree);

    psp_dump_points(hn);

    while (1) {
        long x1[DIM];
        double xd1[DIM];
        for (int i = 0; i < DIM; i++) {
            int a;
            fscanf(stdin, "%d", &a);
            x1[i] = a;
            xd1[i] = a;
        }
        struct svm_node node = { DIM, xd1 };
        PSP_KdSVMTree curr = tree;
        while (curr->node.left && curr->node.right) {
            if (svm_predict(curr->data.model, &node) > 0) {
                curr = (PSP_KdSVMTree)curr->node.left;
            } else {
                curr = (PSP_KdSVMTree)curr->node.right;
            }
        }
        size_t predicted = curr->data.pattern;
        size_t actual = sampl(NULL, x1);
        fprintf(stdout, "%ld -> %ld\n", actual, predicted);
    }
}
