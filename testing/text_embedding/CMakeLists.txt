set(TEST_NAME text_embedding)

add_executable(${TEST_NAME}_accuracy
    $<TARGET_OBJECTS:test_main>
    test_onnx_accuracy.cpp
)
target_link_libraries(${TEST_NAME}_accuracy
    logger
    text_embedding
    gtest
)
set_target_properties(${TEST_NAME}_accuracy PROPERTIES
    BUILD_RPATH "$ORIGIN/../../lib"
    INSTALL_RPATH "$ORIGIN/../../lib"
)
install(TARGETS ${TEST_NAME}_accuracy DESTINATION bin)
add_test(NAME ${TEST_NAME}_accuracy_run COMMAND ${TEST_NAME}_accuracy)

add_executable(${TEST_NAME}_benchmark
    $<TARGET_OBJECTS:test_main>
    test_onnx_benchmark.cpp
)
target_link_libraries(${TEST_NAME}_benchmark
    logger
    text_embedding
    gtest
)
set_target_properties(${TEST_NAME}_benchmark PROPERTIES
    BUILD_RPATH "$ORIGIN/../../lib"
    INSTALL_RPATH "$ORIGIN/../../lib"
)
install(TARGETS ${TEST_NAME}_benchmark DESTINATION bin)
add_test(NAME ${TEST_NAME}_benchmark_run COMMAND ${TEST_NAME}_benchmark)

# === 拷贝脚本文件（确保 Python 测试脚本可用）===
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_onnx_embedding.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/scripts)

# 安装 Python 脚本到 bin/scripts 下
install(DIRECTORY scripts/
    DESTINATION bin/scripts
    FILES_MATCHING PATTERN "*.py"
)
