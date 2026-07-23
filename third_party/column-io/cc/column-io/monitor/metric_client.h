#ifndef RECIS_MONITOR_METRIC_CLIENT_H
#define RECIS_MONITOR_METRIC_CLIENT_H
#pragma once

#include <memory.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#ifdef TORCH_EXTENSION_NAME
#include <torch/custom_class.h>
#endif
#include "column-io/monitor/metric_estimator.h"

namespace recis {
namespace monitor {

using std::map;
using std::mutex;
using std::pair;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
namespace chrono = std::chrono;

class Factory;
class Client;
class PointTag;
enum PointType;  // Option: kGauge, kGaugeSticky, kCounter, kSummary
class RecisMonitorWriter;

/**
 * PointTag: the tag for metric point
 *  Reserved tag key: __* (E.g. __TP, __APPID, __TASKID, __CLUSTER, __QUEUE,
 * __IMAGE, ...)
 */
class PointTag {
 public:
  PointTag(const map<string, string>& kvpairs);
  PointTag(const PointTag& other);
  PointTag& operator=(const PointTag& other);
  bool operator==(const PointTag& other) const;
  bool operator!=(const PointTag& other) const;
  bool operator<(const PointTag& other) const;
  bool operator>(const PointTag& other) const { return other < *this; };

  string to_string();
  static shared_ptr<PointTag> Clone(
      const shared_ptr<PointTag>& other,
      const map<string, string>& kvpairs = map<string, string>());

 private:
  void GenHash();
  int64_t hashid_[2];     // [fnv1a64, std::hash] double hash collision
  string serialize_str_;  // Usage: 1. calculate hashid 2. dump tag values to
                          // writer
  map<string, string> kvpairs_;
};
struct MetricKeyCompLess {
  bool operator()(const pair<string, shared_ptr<PointTag>>& lhs,
                  const pair<string, shared_ptr<PointTag>>& rhs) const;
};

#ifdef TORCH_EXTENSION_NAME
class Factory : public torch::CustomClassHolder {
#else
class Factory {
#endif
  constexpr static chrono::seconds SnapInterval = chrono::seconds(10);

 public:
  Factory(const string& base_path = string(),
          chrono::seconds interval = SnapInterval);
  ~Factory();
  Factory(const Factory&) = delete;
  Factory& operator=(const Factory&) = delete;

  /* * Instance(...) will return an internal static instance(lifecycle is
   * forever)
   * * MakeInstance(...) will give a new one(lifecycle is caller's unique_ptr)
   * Both method will return a workaround instance, you could choose one by your
   * need
   */
  static unique_ptr<Factory>& StaticInstance();
  static unique_ptr<Factory> MakeInstance(
      const string& base_path = string(),
      chrono::seconds interval = SnapInterval);
  static unique_ptr<Factory> MakeInstance(const string& base_path) {
    return MakeInstance(base_path, SnapInterval);
  };
  static unique_ptr<Factory> MakeInstance(chrono::seconds interval) {
    return MakeInstance(string(), interval);
  };
  // static unique_ptr<Factory> MakeInstance(Writer *writer); //TODO: impl with
  // optional writer
  shared_ptr<Client> get_client(const string& client_name);

#ifdef TORCH_EXTENSION_NAME
  static c10::intrusive_ptr<Factory> MakeInstancePy(const string& base_path) {
    return c10::make_intrusive<Factory>(base_path);
  };
  c10::intrusive_ptr<Client> get_client_py(const string& client_name);
#endif

 private:
  void DaemonThreadLaunch();
  void DaemonThread();
  void DaemonThreadJoin();

  mutex cv_mutex_;
  std::condition_variable cv_;
  std::atomic<bool> running_ = {true};
  std::thread thread_;
  chrono::seconds interval_;
  map<string, shared_ptr<Client>> clients_;
#ifdef TORCH_EXTENSION_NAME
  map<string, c10::intrusive_ptr<Client>> clients_py_;
#endif
  unique_ptr<RecisMonitorWriter> writer_;
};

/**
 * Client: the instance for report、statistic and dump metric data. Client
 report is thread-safe
 * Client is managed by a factory, which means it's lifecycle is within the
 factory's lifecycle
 * Every Client has a unique thread to staticstic every point, so that's not
 thread-safe:
  1. client(recis.foo).report(bar)
  2. client(recis).report(foo.bar)
  Be warn of that!
 * NEED NOT close client explicitly, it's more recommended by closing factory,
 to avoid data missing
 */
#ifdef TORCH_EXTENSION_NAME
class Client : public torch::CustomClassHolder {
#else
class Client {
#endif
 public:
  Client(const string& client_name);
  ~Client();
  // thread-safe
  bool report(const string& name, double value,
              const shared_ptr<PointTag>& pTag,
              PointType pType = PointType::kGauge);
  // thread-safe...
  template <typename T>
  typename std::enable_if<std::is_arithmetic<T>::value, bool>::type report(
      const string& name, T value, const shared_ptr<PointTag>& pTag,
      PointType pType = PointType::kGauge) {
    return report(name, static_cast<double>(value), pTag, pType);
  }
  bool reset_metric(const string& name, const shared_ptr<PointTag>& pTag);
  // thread-safe. Be warn of that! some type of point will be cleared after call
  // this func
  string take_snapshot();
#ifdef TORCH_EXTENSION_NAME
  // thread-safe
  bool report_py(const string& name, double value,
                 const c10::Dict<std::string, std::string>& tag_dict,
                 int64_t pType) {
    std::map<std::string, std::string> map_kvpairs;
    for (const auto& item : tag_dict) {
      map_kvpairs[item.key()] = item.value();
    }
    std::shared_ptr<recis::monitor::PointTag> pTag =
        std::make_shared<recis::monitor::PointTag>(map_kvpairs);
    return report(name, value, pTag, static_cast<PointType>(pType));
  }
  bool reset_metric_py(const string& name,
                       const c10::Dict<std::string, std::string>& tag_dict) {
    std::map<std::string, std::string> map_kvpairs;
    for (const auto& item : tag_dict) {
      map_kvpairs[item.key()] = item.value();
    }
    std::shared_ptr<recis::monitor::PointTag> pTag =
        std::make_shared<recis::monitor::PointTag>(map_kvpairs);
    return reset_metric(name, pTag);
  };
#endif

 private:
  string client_name_;
  std::atomic<bool> running_ = {true};
  mutex value_mutex_;  // protect internal data structures

  map<pair<string, shared_ptr<PointTag>>, unique_ptr<Estimator>,
      MetricKeyCompLess>
      estimators_;
};

}  // namespace monitor
}  // namespace recis

#endif  // RECIS_MONITOR_METRIC_CLIENT_H