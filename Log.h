#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

#pragma warning(disable:4996)

/**
 *  Quick and dirty console logging utility.
 */

extern std::string currentISO8601TimeUTC();

#define LOG_CRITICAL
#define LOG_ERROR
#define LOG_WARNING
//#define LOG_DEBUG_1
//#define LOG_DEBUG_2
//#define LOG_DEBUG_3

#ifdef __linux__ 
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#elif _WIN32
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#endif

#define LOG(str) std::cout << "[" << currentISO8601TimeUTC() << "]      " << str << "\n"

#ifdef LOG_CRITICAL 
#define CRIT(str) std::cout << "[" << currentISO8601TimeUTC() << "] CRIT " << str << " <" << __FILENAME__ << "> (" << __func__ << ":" << __LINE__ << ")\n"
#else
#define CRIT(str)
#endif

#ifdef LOG_ERROR 
#define ERR(str) std::cout << "[" << currentISO8601TimeUTC() << "] ERR  " << str << " <" << __FILENAME__ << "> (" << __func__ << ":" << __LINE__ << ")\n"
#else
#define ERR(str)
#endif

#ifdef LOG_WARNING 
#define WARN(str) std::cout << "[" << currentISO8601TimeUTC() << "] WARN " << str << " <" << __FILENAME__ << "> (" << __func__ << ":" << __LINE__ << ")\n"
#else
#define WARN(str)
#endif

#ifdef LOG_DEBUG_1 
#define LOG1(str) std::cout << "[" << currentISO8601TimeUTC() << "] dgb1 " << str << " <" << __FILENAME__ << "> (" << __func__ << ":" << __LINE__ << ")\n"
#else
#define LOG1(str)
#endif

#ifdef LOG_DEBUG_2 
#define LOG2(str) std::cout << "[" << currentISO8601TimeUTC() << "] dgb2 " << str << " <" << __FILENAME__ << "> (" << __func__ << ":" << __LINE__ << ")\n"
#else
#define LOG2(str)
#endif

#ifdef LOG_DEBUG_3 
#define LOG3(str) std::cout << "[" << currentISO8601TimeUTC() << "] dgb3 " << str << " <" << __FILENAME__ << "> (" << __func__ << ":" << __LINE__ << ")\n"
#else
#define LOG3(str)
#endif
