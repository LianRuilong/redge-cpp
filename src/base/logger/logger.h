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

class LogMessage {
public:
    explicit LogMessage(LogLevel level);
    ~LogMessage();

    template <typename T>
    LogMessage& operator<<(const T& msg) {
        stream_ << msg;
        return *this;
    }

private:
    LogLevel level_;
    std::ostringstream stream_;
};

// 初始化和关闭接口
void InitLogger(const char* program_name, int stderr_level = 0);
void ShutdownLogger();

}  // namespace logger

// === 宏封装 ===
#define LOG_INFO    logger::LogMessage(logger::LogLevel::INFO)
#define LOG_WARNING logger::LogMessage(logger::LogLevel::WARNING)
#define LOG_ERROR   logger::LogMessage(logger::LogLevel::ERROR)
#define LOG_FATAL   logger::LogMessage(logger::LogLevel::FATAL)

#ifdef ENABLE_DEBUG_LOG
#define LOG_DEBUG LOG_INFO
#else
#define LOG_DEBUG while(false) std::cerr
#endif
