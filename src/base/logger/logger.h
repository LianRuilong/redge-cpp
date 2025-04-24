#pragma once

#include <sstream>
#include <string>

namespace logger {

enum class LogLevel {
    INFO,
    WARNING,
    ERROR,
    FATAL
};

class LoggerStream {
public:
    LoggerStream(LogLevel level, const char* file, int line);
    ~LoggerStream();

    template <typename T>
    LoggerStream& operator<<(const T& msg) {
        stream_ << msg;
        return *this;
    }

private:
    LogLevel level_;
    const char* file_;
    int line_;
    std::ostringstream stream_;
};

// 初始化和关闭接口
void InitLogger(const char* program_name, int stderr_level = 0);
void ShutdownLogger();

}  // namespace logger

// === 宏封装 ===
#define LOG_INFO    logger::LoggerStream(logger::LogLevel::INFO,    __FILE__, __LINE__)
#define LOG_WARNING logger::LoggerStream(logger::LogLevel::WARNING, __FILE__, __LINE__)
#define LOG_ERROR   logger::LoggerStream(logger::LogLevel::ERROR,   __FILE__, __LINE__)
#define LOG_FATAL   logger::LoggerStream(logger::LogLevel::FATAL,   __FILE__, __LINE__)

#ifdef ENABLE_DEBUG_LOG
#define LOG_DEBUG logger::LoggerStream(logger::LogLevel::INFO, __FILE__, __LINE__)
#else
#define LOG_DEBUG while(false) logger::LoggerStream(logger::LogLevel::INFO, __FILE__, __LINE__)
#endif
