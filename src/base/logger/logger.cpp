#include "logger.h"
#include <glog/logging.h>

namespace logger {

LoggerStream::LoggerStream(LogLevel level, const char* file, int line)
    : level_(level), file_(file), line_(line) {}

LoggerStream::~LoggerStream() {
    switch (level_) {
        case LogLevel::INFO:
            google::LogMessage(file_, line_, google::GLOG_INFO).stream() << stream_.str();
            break;
        case LogLevel::WARNING:
            google::LogMessage(file_, line_, google::GLOG_WARNING).stream() << stream_.str();
            break;
        case LogLevel::ERROR:
            google::LogMessage(file_, line_, google::GLOG_ERROR).stream() << stream_.str();
            break;
        case LogLevel::FATAL:
            google::LogMessage(file_, line_, google::GLOG_FATAL).stream() << stream_.str();
            break;
    }
}

void InitLogger(const char* program_name, int stderr_level) {
    google::InitGoogleLogging(program_name);
    google::SetStderrLogging(static_cast<google::LogSeverity>(stderr_level));
    google::InstallFailureSignalHandler();
}

void ShutdownLogger() {
    google::ShutdownGoogleLogging();
}

}  // namespace logger
