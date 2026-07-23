#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "column-io/monitor/log_writer.h"
#include "column-io/monitor/sock_writer.h"
}

#include "column-io/monitor/metric_client.h"

namespace recis {
namespace monitor {

namespace chrono = std::chrono;

namespace {
static int64_t __recis_fnv1a64(const string& s) {
  int64_t hash = 0xcbf29ce484222325;

  const char* str = s.c_str();
  for (int c; (c = *str++);) {
    hash ^= c;
    hash *= 0x100000001b3;
  }
  return hash;
}

}  // namespace

/* -------- PointTag -------- */
PointTag::PointTag(const map<string, string>& kvpairs) : kvpairs_(kvpairs) {
  GenHash();
}
PointTag::PointTag(const PointTag& other) : kvpairs_(other.kvpairs_) {
  GenHash();
}
PointTag& PointTag::operator=(const PointTag& other) {
  if (this == &other) return *this;
  kvpairs_ = other.kvpairs_;  // TODO: this copy is slow, need to optimize out
                              // if Clone() method neednot
  GenHash();
  return *this;
}
bool PointTag::operator==(const PointTag& other) const {
  if (this == &other) return true;
  return hashid_[0] == other.hashid_[0] && hashid_[1] == other.hashid_[1];
}
bool PointTag::operator!=(const PointTag& other) const {
  if (this == &other) return false;
  return hashid_[0] != other.hashid_[0] && hashid_[1] != other.hashid_[1];
}
bool PointTag::operator<(const PointTag& other) const {
  if (this == &other) return true;
  if (hashid_[0] != other.hashid_[0]) return hashid_[0] < other.hashid_[0];
  return hashid_[1] < other.hashid_[1];
}
shared_ptr<PointTag> PointTag::Clone(const shared_ptr<PointTag>& other,
                                     const map<string, string>& kvpairs) {
  map<string, string> merged =
      (other ? other->kvpairs_ : map<string, string>{});
  for (auto& kv : kvpairs) merged[kv.first] = kv.second;

  return std::make_shared<PointTag>(merged);
}
void PointTag::GenHash() {
  vector<pair<string, string>> kvpairs_vec;
  kvpairs_vec.insert(kvpairs_vec.end(), kvpairs_.begin(), kvpairs_.end());
  std::sort(kvpairs_vec.begin(), kvpairs_vec.end());
  std::ostringstream oss;
  for (size_t i = 0; i < kvpairs_vec.size(); ++i) {
    if (i != 0) oss << ',';
    oss << kvpairs_vec[i].first << '=' << kvpairs_vec[i].second;
  }
  serialize_str_ = oss.str();
  hashid_[0] = __recis_fnv1a64(serialize_str_);
  hashid_[1] = std::hash<string>{}(serialize_str_);
  // });
}
string PointTag::to_string() { return serialize_str_; }
bool MetricKeyCompLess::operator()(
    const pair<string, shared_ptr<PointTag>>& lhs,
    const pair<string, shared_ptr<PointTag>>& rhs) const {
  // Different metric name use different estimator, apparently
  if (lhs.first < rhs.first) return true;
  if (rhs.first < lhs.first) return false;
  // Same metric name, but different tag, also use different estimator
  if (lhs.second == rhs.second) return false;
  if (!lhs.second) return true;
  if (!rhs.second) return false;
  return *(lhs.second) < *(rhs.second);
};

/* -------- Estimator -------- */
unique_ptr<Estimator> Estimator::NewEstimator(PointType pType) {
  unique_ptr<Estimator> estimator_p_;
  switch (pType) {
    case PointType::kGauge:
    case PointType::kGaugeSticky:
    case PointType::kCounter: {
      estimator_p_ = unique_ptr<Estimator>(new TrivialEstimator());
      break;
    }
    case PointType::kSummary: {
      estimator_p_ = unique_ptr<Estimator>(new BucketizeEstimator());
      break;
    }
    default: {
      throw std::runtime_error("Metric Estimator unsupported PointType: " +
                               std::to_string(pType));
    }
  }
  estimator_p_->pType = pType;
  return estimator_p_;
}

/* -------- Writer -------- */
class RecisMonitorWriter {
 public:
  enum WriteType : int32_t {
    kEmpty = 0,
    kUDS = 1,
    kLog = 2,
    kInvalid = INT32_MAX
  };
  RecisMonitorWriter(const char* base_path);
  ~RecisMonitorWriter();
  int write(const char* msg);

 private:
  WriteType writer_type_;
  unique_ptr<RECIS_MONITOR_LOG_WRITER_T> log_writer_;
  unique_ptr<RECIS_MONITOR_SOCK_WRITER> sock_writer_;
};
RecisMonitorWriter::RecisMonitorWriter(const char* base_path) {
  const char* writer_type_s_ = std::getenv("RECIS_MONITOR_WRITER");
  writer_type_ =
      static_cast<WriteType>(atoi(writer_type_s_ ? writer_type_s_ : "1"));
  std::string writer_logname(base_path);
  std::string default_logname = "./log/";
  if (std::getenv("STD_LOG_DIR")) {
    default_logname = std::getenv("STD_LOG_DIR");
  }
  if (default_logname.back() != '/') {
    default_logname += '/';
  }
  switch (writer_type_) {
    case RecisMonitorWriter::WriteType::kEmpty:
      break;
    case RecisMonitorWriter::WriteType::kUDS: {
      if (writer_logname.empty()) {
        writer_logname = default_logname + "/recis_metric.sock";
        // E.g. ./log/recis_metric.sock
      }
      sock_writer_.reset(new RECIS_MONITOR_SOCK_WRITER(writer_logname.c_str()));
      break;
    }
    case RecisMonitorWriter::WriteType::kLog: {
      if (writer_logname.empty()) {
        default_logname += (std::getenv("RANK") ? std::getenv("RANK") : "0");
        writer_logname = default_logname + "/recis_metric.log";
        // E.g. ./log/0/recis_metric.log
      }
      log_writer_.reset(new RECIS_MONITOR_LOG_WRITER_T());
      int ret = recis_monitor_init_logger(
          log_writer_.get(), writer_logname.c_str(),
          RECIS_MONITOR_LOG_ROTATE_T::RECIS_MONITOR_LOG_ROTATE_DEFAULT);
      if (ret != 0) {
        throw std::runtime_error(
            "Metric Writer construct error, writer_ init ret: " +
            std::to_string(ret));
      }
      break;
    }
    default:
      break;
  }
}
RecisMonitorWriter::~RecisMonitorWriter() {
  switch (writer_type_) {
    case RecisMonitorWriter::WriteType::kEmpty:
      return;
    case RecisMonitorWriter::WriteType::kUDS: {
      sock_writer_.reset();
      return;
    }
    case RecisMonitorWriter::WriteType::kLog: {
      recis_monitor_close_logger(log_writer_.get());
      log_writer_.reset();
      return;
    }
    default:
      return;
  }
}
int RecisMonitorWriter::write(const char* msg) {
  switch (writer_type_) {
    case RecisMonitorWriter::WriteType::kEmpty: {
      return 0;
    }
    case RecisMonitorWriter::WriteType::kUDS: {
      return sock_writer_->write(msg);
    }
    case RecisMonitorWriter::WriteType::kLog: {
      return recis_monitor_logger_write(log_writer_.get(), msg, true);
    }
    default: {
      return -1;
    }
  }
}

/* -------- Factory -------- */
Factory::Factory(const string& base_path, chrono::seconds interval)
    : interval_(interval) {
  writer_.reset(new RecisMonitorWriter(base_path.c_str()));
  DaemonThreadLaunch();
}
Factory::~Factory() {
  DaemonThreadJoin();

  {
    std::lock_guard<mutex> lg(cv_mutex_);
    for (map<string, shared_ptr<Client>>::iterator it = clients_.begin();
         it != clients_.end(); ++it) {
      it->second.reset();
    }
    clients_.clear();
#ifdef TORCH_EXTENSION_NAME
    for (map<string, c10::intrusive_ptr<Client>>::iterator it =
             clients_py_.begin();
         it != clients_py_.end(); ++it) {
      it->second.reset();
    }
    clients_py_.clear();
#endif
  }
  writer_.reset();
}
void Factory::DaemonThreadLaunch() {
  if (!running_.load(std::memory_order_seq_cst)) {
    return;
  }
  thread_ = std::thread(&Factory::DaemonThread, this);
}
void Factory::DaemonThread() {
  chrono::system_clock::time_point now = chrono::system_clock::now();
  chrono::system_clock::time_point last_snapshot_ =
      now - now.time_since_epoch() % interval_ + interval_;

  // TODO: setconf(vec<name, val, default>),  set __LANGUAGE=python/cxx
  string global_tag = "__Framework=recis";
  global_tag += ",__Pid=" + std::to_string(getpid());
  global_tag +=
      ",__FactInst=" + std::to_string(reinterpret_cast<uintptr_t>(this));

  string rank = (std::getenv("RANK")) ? std::getenv("RANK") : "0";
  string world_size =
      (std::getenv("WORLD_SIZE")) ? std::getenv("WORLD_SIZE") : "1";
  string task_name =
      (std::getenv("TASK_NAME")) ? std::getenv("TASK_NAME") : "worker";
  global_tag += ",__Rank=" + rank + ",__WorldSize=" + world_size +
                ",__TaskName=" + task_name;

  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(6);
  for (std::unique_lock<mutex> lk(cv_mutex_); running_.load();
       last_snapshot_ += interval_) {
    cv_.wait_until(lk, last_snapshot_, [this] { return !running_.load(); });
    if (!running_.load()) {
      break;
    }
    oss.str(string{});
    chrono::milliseconds ms_now = chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch());
    oss << "\n# BeginGroup recis.monitor.factory," << global_tag << "\t";
    oss << std::to_string(ms_now.count()) << "\n";
    {
      for (auto& it : clients_) {
        shared_ptr<Client>& client = it.second;
        string snapshot = client->take_snapshot();
        oss << snapshot;
      }
#ifdef TORCH_EXTENSION_NAME
      for (auto& it : clients_py_) {
        c10::intrusive_ptr<Client>& client = it.second;
        string snapshot = client->take_snapshot();
        oss << snapshot;
      }
#endif
    }
    oss << "# EndGroup recis.monitor.factory\t";
    oss << std::to_string(ms_now.count()) << "\n";
    lk.unlock();  // avoiding deadlock from timeout or writer_ block
    writer_->write(oss.str().c_str());
    lk.lock();
  }
  // return; Release cv&mutex to join DaemonThread
}
void Factory::DaemonThreadJoin() {
  {
    std::lock_guard<std::mutex> lg(cv_mutex_);
    running_.store(false);
  }
  cv_.notify_all();  // 唤醒wait_until提前退出
  if (thread_.joinable()) {
    thread_.join();
  }
}
unique_ptr<Factory>& Factory::StaticInstance() {
  static unique_ptr<Factory> static_instance;
  if (__builtin_expect(!static_instance, false)) {
    static_instance.reset((new Factory()));
  }
  return static_instance;
}
unique_ptr<Factory> Factory::MakeInstance(const string& base_path,
                                          chrono::seconds interval) {
  return unique_ptr<Factory>(new Factory(base_path, interval));
}
shared_ptr<Client> Factory::get_client(const string& client_name) {
  std::lock_guard<mutex> lg(cv_mutex_);
  map<string, shared_ptr<Client>>::iterator it = clients_.find(client_name);
  if (it != clients_.end()) return it->second;
  shared_ptr<Client> cli = std::make_shared<Client>(client_name);
  clients_.emplace(client_name, cli);
  return cli;
}
#ifdef TORCH_EXTENSION_NAME
c10::intrusive_ptr<Client> Factory::get_client_py(const string& client_name) {
  std::lock_guard<mutex> lg(cv_mutex_);
  map<string, c10::intrusive_ptr<Client>>::iterator it =
      clients_py_.find(client_name);
  if (it != clients_py_.end()) return it->second;
  c10::intrusive_ptr<Client> cli = c10::make_intrusive<Client>(client_name);
  clients_py_.emplace(client_name, cli);
  return cli;
}
#endif

/* -------- Client -------- */
Client::Client(const string& client_name) : client_name_(client_name) {}
Client::~Client() {
  if (!running_.load()) return;
  std::lock_guard<mutex> lg(value_mutex_);
  running_.store(false);
  estimators_.clear();
}
bool Client::report(const string& name, double value,
                    const shared_ptr<PointTag>& pTag, PointType pType) {
  std::lock_guard<mutex> lg(value_mutex_);
  if (__builtin_expect(!running_.load(), false)) {
    return false;
  }

  pair<string, shared_ptr<PointTag>> key = std::make_pair(name, pTag);
  if (estimators_.find(key) == estimators_.end()) {
    estimators_[key] = Estimator::NewEstimator(pType);
  };

  Estimator& estimator = *(estimators_[key]);
  if (!estimator.checkpass_type(pType)) {
    return false;
  }

  switch (pType) {
    case PointType::kGauge:
    case PointType::kGaugeSticky:
    case PointType::kCounter: {
      estimator.observe(value);
      break;
    }
    // 为了灵活切换metric后端、避免耦合,本处仅实现单机Summary方案,不考虑分布式分位数的聚合问题
    case PointType::kSummary: {
      estimator.observe(static_cast<double>(value));
      break;
    }
    default: {
      throw std::runtime_error("Metric Client unsupported PointType:" +
                               std::to_string(pType));
      break;
    }
  }
  return true;
}
bool Client::reset_metric(const string& name,
                          const shared_ptr<PointTag>& pTag) {
  std::lock_guard<mutex> lg(value_mutex_);
  if (__builtin_expect(!running_.load(), false)) {
    return true;
  }

  pair<string, shared_ptr<PointTag>> key = std::make_pair(name, pTag);
  if (estimators_.find(key) == estimators_.end()) {
    return true;
  }
  Estimator& estimator = *(estimators_[key]);
  estimator.reset();
  return true;
}
string Client::take_snapshot() {
  std::lock_guard<mutex> lg(value_mutex_);
  if (__builtin_expect(!running_.load(), false)) {
    return string();
  }

  auto func_get_type_name = [](PointType pType) -> const char* {
    switch (pType) {
      case PointType::kGauge:
      case PointType::kGaugeSticky:
        // 驻留型和非驻留型点的记录时有区别但上报类型都是gauge, 无需区分
        return "gauge";
      case PointType::kCounter:
        return "count";
      case PointType::kSummary:
        return "sumry";
      // case PointType::kHistogram: return "histg";
      // case PointType::kHistogramFull: return "histf";
      default:
        return "gauge";
    }
  };

  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(6);

  for (auto& it : estimators_) {
    const string& name = it.first.first;
    const shared_ptr<PointTag>& pTag = it.first.second;
    Estimator& estimator = *(it.second);
    if (estimator.count() == 0 && estimator.pType != PointType::kCounter) {
      continue;
    }
    oss << client_name_ << "." << name
        << ",__TP=" << func_get_type_name(estimator.pType);
    if (pTag.get() && !pTag->to_string().empty()) {
      oss << "," << pTag->to_string();
    }
    oss << '\t';
    switch (estimator.pType) {
      case PointType::kGauge:
      case PointType::kGaugeSticky: {
        double val = estimator.last();     // Original Value of PointType
        uint64_t num = estimator.count();  // Original Point Number of PointType
        double sum = estimator.sum();
        double avg = estimator.sum() / static_cast<double>(estimator.count());
        double max = estimator.max();
        double min = estimator.min();
        oss << "val=" << val << ',' << "num=" << num << ',' << "sum=" << sum
            << ',' << "avg=" << avg << ',' << "max=" << max << ','
            << "min=" << min;
        if (estimator.pType == PointType::kGauge) {
          estimator.reset();
        }
        break;
      }
      case PointType::kCounter: {
        double val = estimator.sum();      // Original Value of PointType
        uint64_t num = estimator.count();  // Original Point Number of PointType
        oss << "val=" << val << ',' << "num=" << num;
        estimator.reset();
        break;
      }
      case PointType::kSummary: {
        double p10 = estimator.query(0.10);
        double p50 = estimator.query(0.50);
        double p90 = estimator.query(0.90);
        double p99 = estimator.query(0.99);
        oss << "p10=" << p10 << ',' << "p50=" << p50 << ',' << "p90=" << p90
            << ',' << "p99=" << p99;
        double avg = estimator.sum() / static_cast<double>(estimator.count());
        double max = estimator.max();
        oss << ',' << "avg=" << avg << ',' << "max=" << max;
        estimator.reset();
        break;
      }
      default: {
        // oss << "## ERROR Unsupport Client:" << client_name_ << ", Point:" <<
        // name << ", Type:" << estimator.pType << '\n';
        oss << "inv=nan";
        break;
      }
    }

    int64_t timestamp_ms = chrono::duration_cast<chrono::milliseconds>(
                               chrono::system_clock::now().time_since_epoch())
                               .count();
    oss << "\t" << std::to_string(timestamp_ms) << '\n';
  }
  return oss.str();
}

}  // namespace monitor
}  // namespace recis