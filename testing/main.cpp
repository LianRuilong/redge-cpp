#include <gtest/gtest.h>

#include "logger.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // 使用封装好的初始化函数
    logger::InitLogger(argv[0], 0);  // 0 表示 INFO 等级
    int ret = RUN_ALL_TESTS();

    logger::ShutdownLogger();  // 清理日志资源
    return ret;
}