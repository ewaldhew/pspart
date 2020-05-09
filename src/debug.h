#ifndef DEBUG_H
#define DEBUG_H

#if DEBUG
#include <iostream>
#define DEBUG_LOG(str) do { std::cerr << str; } while (false)
#else
#define DEBUG_LOG(str) do { } while (false)
#endif

#endif

/* EOF */
