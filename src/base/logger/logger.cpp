#include "logger.h"
#include <glog/logging.h>

namespace logger {

LogMessage::LogMessage(LogLevel level) : level_(level) {}

LogMessage::~LogMessage() {
    switch (level_) {
        case LogLevel::INFO:
            LOG(INFO) << stream_.str();
            break;
        case LogLevel::WARNING:
            LOG(WARNING) << stream_.str();
            break;
        case LogLevel::ERROR:
            LOG(ERROR) << stream_.str();
            break;
        case LogLevel::FATAL:
            LOG(FATAL) << stream_.str();
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
