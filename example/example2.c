#include <stdio.h>
#include <pspart.h>


size_t sampl(void* sc, long* pnt)
{
    size_t dec = 0;
    for (int i = 0; i < 4; i++) {
        dec |= (pnt[i] < 0 ? 0 : 1)  << i;
    }
    return dec;
}

int main()
{
    PSP_Handle hn = PSP_New(4);
    PSP_Sampling_CallbackRec cb = {hn, sampl};
    Fixed x0[8] = { 0,0,0,0, 65536,65536,65536,65536 };
    Fixed xm[4] = { -65536,-65536,-65536,-65536 };
    Fixed xM[4] = { 65536,65536,65536,65536 };
    PSP_Options options = {0};
    PSP_Get_Regions(hn, &cb, 2, x0, xm, xM, options, PSP_RESULT_APPEND);
    PSP_KdSVMTree tree = NULL;
    PSP_Build_Partition_KdSVM(hn, &tree);

    long offset = -65000;
    for (long x = -65536; x <= 65536; x++) {
        long x1[4] = { x, offset, 0, 0 };
        double xd1[4] = { x, offset, 0, 0 };
        struct svm_node node = { 4, xd1 };
        PSP_KdSVMTree curr = tree;
        while (curr->node.left && curr->node.right) {
            if (svm_predict(curr->data.model, &node) < 0) {
                curr = (PSP_KdSVMTree)curr->node.left;
            } else {
                curr = (PSP_KdSVMTree)curr->node.right;
            }
        }
        size_t predicted = curr->data.pattern;
        size_t actual = sampl(NULL, x1);
        if (predicted != actual) {
            fprintf(stdout, "point x = %ld failed, %ld->%ld\n", x, actual, predicted);
        }

        long x2[4] = { offset, x, 0, 0 };
        double xd2[4] = { offset, x, 0, 0 };
        node.values = xd2;
        curr = tree;
        while (curr->node.left && curr->node.right) {
            if (svm_predict(curr->data.model, &node) < 0) {
                curr = (PSP_KdSVMTree)curr->node.left;
            } else {
                curr = (PSP_KdSVMTree)curr->node.right;
            }
        }
        predicted = curr->data.pattern;
        actual = sampl(NULL, x2);
        if (predicted != actual)
            fprintf(stdout, "point y = %ld failed, %ld->%ld\n", x, actual, predicted);
    }
}
