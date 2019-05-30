#ifndef DEBUG_H
#define DEBUG_H

#ifdef DEBUG
#include <iostream>
#define DEBUG_LOG(str) do { std::cout << str; } while (false)
#else
#define DEBUG_LOG(str) do { } while (false)
#endif

#endif

/* EOF */
