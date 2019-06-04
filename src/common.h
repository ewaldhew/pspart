#ifndef COMMON_H_
#define COMMON_H_

#ifdef __cplusplus
#include <memory>


struct Node_Internal {
    using Node_InternalPtr = std::shared_ptr<Node_Internal>;
    Node_InternalPtr left;
    Node_InternalPtr right;

    Node_Internal(Node_InternalPtr left, Node_InternalPtr right)
    : left(left), right(right) { };
    Node_Internal(Node_Internal const& other) = delete;
    Node_Internal(Node_Internal && other) = delete;
    Node_Internal & operator=(Node_Internal const& other) = delete;
    Node_Internal & operator=(Node_Internal && other) = delete;
    ~Node_Internal() = default;
};
using Node_InternalPtr = std::shared_ptr<Node_Internal>;

typedef struct PSP_MemoryRec {
    Node_InternalPtr kdsvm;
    Node_InternalPtr mcsvm;
} *PSP_Memory;

extern "C"
{
#endif

typedef struct PSP_NodeRec_ PSP_NodeRec, *PSP_Node;
struct PSP_NodeRec_ {
    PSP_Node left;
    PSP_Node right;
};

#ifdef __cplusplus
}
#endif

#endif

/* EOF */
