# format cpp files with google style
echo "Format all CPP files"
find $PWD/csrc -type f \( \
  -name '*.h' -or \
  -name '*.hpp' -or \
  -name '*.cpp' -or \
  -name '*.cu' -or \
  -name '*.cuh' -or \
  -name '*.c' -or \
  -name '*.cc' \)| \
  xargs clang-format -style=file --sort-includes -i


echo "Format all python files" 
isort $PWD && ruff format $PWD
