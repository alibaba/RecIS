find cc -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*cc" | xargs clang-format -i
cd column_io; black .