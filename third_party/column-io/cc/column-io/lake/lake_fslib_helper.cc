#include "lake_fslib_helper.h"
#include <functional>
namespace lake {
bool getListDir(const std::string &dir, std::vector<std::string> *result) {
  void *libHandle = dlopen(getenv("LAKERUNTIMEso"), RTLD_LAZY | RTLD_LOCAL);
  if (!libHandle) {
    return false;
  }
  void *symbol_ptr = nullptr;
  symbol_ptr = dlsym(libHandle, "getListDir");
  if (!symbol_ptr) {
    return false;
  }
  char **temp_list;
  size_t size;
  std::function<bool(const char *, char ***, size_t *)> func =
      reinterpret_cast<bool (*)(const char *, char ***, size_t *)>(symbol_ptr);
  bool flag = func(dir.c_str(), &temp_list, &size);
  if (!flag) {
    return flag;
  }
  result->clear();
  for (int i = 0; i < size; i++) {
    result->push_back(std::string(temp_list[i]));
  }
  return true;
};

bool closePangu() {
  void *libHandle = dlopen(getenv("LAKERUNTIMEso"), RTLD_LAZY | RTLD_LOCAL);
  if (!libHandle) {
    return false;
  }
  void *symbol_ptr = nullptr;
  symbol_ptr = dlsym(libHandle, "closePangu");
  if (!symbol_ptr) {
    return false;
  }
  std::function<void()> func = reinterpret_cast<void (*)()>(symbol_ptr);
  func();
  return true;
}

} // namespace lake
