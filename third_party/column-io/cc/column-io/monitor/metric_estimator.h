#ifndef RECIS_MONITOR_METRIC_ESTIMATOR_H
#define RECIS_MONITOR_METRIC_ESTIMATOR_H
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace recis {
namespace monitor {
using std::map;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class Estimator;
class TrivialEstimator;
class CKMSEstimator;
class BucketizeEstimator;

enum PointType {
  kGauge = 100LL,        // 取周期内最后一个值
  kGaugeSticky = 110LL,  // 取周期内外最后一个值, 不自动清理
  kCounter = 200LL,      // 取本client启动后, Tag序列下的累计值
  kSummary = 300LL,      // 取周期内sum/avg/max/p99. kmonitor
                     // sdk中的Summary含义等同于Histogram, 因故合并
  // kHistogram = 400LL,    // 取周期内sum/avg/max/p99
  // kHistogramFull = 410LL,  // 取周期内sum/avg/max/min/p01/p10/p50/p90/p99
  kInvalid = INT64_MAX
};

// 抽象基类
class Estimator {
 public:
  virtual ~Estimator() {}
  virtual void observe(double value) {
    throw std::runtime_error("Estimator::observe() not implemented");
  };
  virtual double query(double quantile) {
    throw std::runtime_error("Estimator::query() not implemented");
  }
  virtual void reset() {
    throw std::runtime_error("Estimator::reset() not implemented");
  }
  virtual uint64_t count() const {
    throw std::runtime_error("Estimator::count() not implemented");
  }
  virtual double sum() const {
    throw std::runtime_error("Estimator::sum() not implemented");
  }
  virtual double max() const {
    throw std::runtime_error("Estimator::max() not implemented");
  }
  virtual double min() const {
    throw std::runtime_error("Estimator::min() not implemented");
  }
  virtual double last() const {
    throw std::runtime_error("Estimator::last() not implemented");
  }
  inline bool checkpass_type(PointType otherType) {
    if (__builtin_expect(otherType == pType, true)) {
      return true;
    }
    return false;
  }
  static unique_ptr<Estimator> NewEstimator(PointType pType);
  PointType pType;
};

/* -------- TrivialEstimator -------- */
class TrivialEstimator : public Estimator {
 public:
  TrivialEstimator() : count_(0), sum_(0), max_(0), min_(0), last_(0) {}
  void observe(double value) override {
    if (!count_ || value > max_) max_ = value;
    if (!count_ || value < min_) min_ = value;
    if (!count_ || value != last_) last_ = value;

    sum_ += value;
    count_++;
  }
  double query(double quantile) override {
    throw std::runtime_error("TrivialEstimator::query() not implemented");
  }
  void reset() override {
    count_ = 0;
    sum_ = 0.0;
    max_ = 0.0;
    min_ = 0.0;
    last_ = 0.0;
  }
  uint64_t count() const override { return count_; }
  double sum() const override { return static_cast<double>(sum_); }
  double max() const override { return static_cast<double>(max_); }
  double min() const override { return static_cast<double>(min_); }
  double last() const override { return static_cast<double>(last_); }

 private:
  uint64_t count_;
  double sum_;
  double max_;
  double min_;
  double last_;
};

/* -------- CKMSEstimator -------- */
// WARNING: DO NOT USE IT NOW! CKMS Estimator IS NOT FULLY IMPLEMENTED
class CKMSEstimator : public Estimator {
 private:
  struct CKMSSample {
    double v;
    int g;
    int delta;
  };

 public:
  CKMSEstimator(double epsilon) : epsilon_(epsilon), n_(0), sum_(0) {}

  void observe(double value) override {
    auto it = std::lower_bound(
        samples_.begin(), samples_.end(), value,
        [](const CKMSSample &s, double val) { return s.v < val; });
    int delta = (it == samples_.begin() || it == samples_.end())
                    ? 0
                    : std::floor(2 * epsilon_ * n_);
    samples_.insert(it, CKMSSample{value, 1, delta});
    ++n_;
    sum_ += value;
    compress();
  }
  double query(double quantile) override {
    if (samples_.empty()) return NAN;
    int rankMin = 0;
    int desired = std::floor(quantile * n_);
    for (size_t i = 0; i < samples_.size(); ++i) {
      rankMin += samples_[i].g;
      int rankMax = rankMin + samples_[i].delta;
      if (rankMax > desired + (epsilon_ * n_)) {
        return samples_[i].v;
      }
    }
    return samples_.back().v;
  }
  void reset() override {
    n_ = 0;
    sum_ = 0;
    samples_.clear();
  }
  uint64_t count() const override { return n_; }
  double sum() const override { return sum_; }
  double max() const override { return NAN; }
  double min() const override { return NAN; }
  double last() const override { return NAN; }

 private:
  void compress() {
    for (size_t i = 0; i + 1 < samples_.size();) {
      if (samples_[i].g + samples_[i + 1].g + samples_[i + 1].delta <=
          std::floor(2 * epsilon_ * n_)) {
        samples_[i + 1].g += samples_[i].g;
        samples_.erase(samples_.begin() + i);
      } else {
        ++i;
      }
    }
  }

  double epsilon_;
  uint64_t n_;
  double sum_;
  vector<CKMSSample> samples_;
};

/* -------- BucketizeEstimator -------- */
// A Simple Bucket Struct for quantile estimation
const static size_t Bucket_kswitchThreshold = 48;
const static double Bucket_kAlpha = 0.25;
// 1.25^48 ~= 44814. [1ms, 44s) enough range for uniform-distribution ops

class SimpleBucket {
 public:
  // SmartBucket(size_t switchThreshold = Bucket_kswitchThreshold) :
  // threshold_(switchThreshold), count_(0), sum_(0) {}
  SimpleBucket() = delete;
  SimpleBucket(const SimpleBucket &other) = delete;
  SimpleBucket(double alpha) noexcept
      : alpha_(alpha), threshold_(Bucket_kswitchThreshold) {};
  SimpleBucket(SimpleBucket &&other) noexcept
      : alpha_(other.alpha_),
        threshold_(other.threshold_),
        vec_(std::move(other.vec_)),
        map_p_(std::move(other.map_p_)),
        count_(other.count_),
        sum_(other.sum_) {
    other.count_ = 0;
    other.sum_ = 0;
  }
  SimpleBucket &operator=(SimpleBucket &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    alpha_ = other.alpha_;
    threshold_ = other.threshold_;
    vec_ = std::move(other.vec_);
    map_p_ = std::move(other.map_p_);
    count_ = other.count_;
    sum_ = other.sum_;
    other.count_ = 0;
    other.sum_ = 0;
    return *this;
  }

 private:
  void switchToMap() {
    if (map_p_ != nullptr) {
      return;
    }
    map_p_ = unique_ptr<map<int, uint64_t>>();
    for (auto &kv : vec_) {
      (*map_p_)[kv.first] = kv.second;
    }
    vec_.resize(0);
  }
  bool is_switched() const { return map_p_ != nullptr; }
  void insertMultiple(int key, uint64_t times) {
    if (map_p_) {
      (*map_p_)[key] += times;
      return;
    }
    for (auto &kv : vec_) {
      if (kv.first == key) {
        kv.second += times;
        return;
      }
    }
    vec_.emplace_back(key, times);
    if (vec_.size() > threshold_) {
      switchToMap();
    }
  }
  double keyToValue(int key, double alpha) const {
    return std::pow(1.0 + alpha, std::abs(key)) * ((key < 0) ? -1 : 1);
  }
  void insert_key(int key) {
    count_++;
    sum_ += keyToValue(key, alpha_);
    if (map_p_) {
      (*map_p_)[key]++;
    } else {
      for (auto &kv : vec_) {
        if (kv.first == key) {
          kv.second++;
          return;
        }
      }
      vec_.emplace_back(key, 1);
      if (vec_.size() > threshold_) {
        switchToMap();
      }
    }
  }

 public:
  void insert(double v) {
    if (v == 0) v = 1e-12;
    int sign = (v < 0) ? -1 : 1;
    double absVal = std::abs(v);
    int key =
        static_cast<int>(std::ceil(std::log(absVal) / std::log(1 + alpha_)));
    key *= sign;
    insert_key(key);
  }

  void merge(const SimpleBucket &other) {
    if (!map_p_ && !other.map_p_) {
      for (auto &kv : other.vec_) {
        insertMultiple(kv.first, kv.second);
      }
    } else if (other.map_p_ != nullptr) {
      switchToMap();
      for (auto &kv : *(other.map_p_)) {
        insertMultiple(kv.first, kv.second);
      }
    } else {
      switchToMap();
      for (auto &kv : other.vec_) {
        insertMultiple(kv.first, kv.second);
      }
    }
    count_ += other.count_;
    sum_ += other.sum_;
  }

  void reset() {
    vec_.resize(0);
    map_p_.reset();
    count_ = 0;
    sum_ = 0.0;
  }

  double query(double quantile) const {
    quantile = std::min(std::max(quantile, 0.0001), 0.9999);
    uint64_t targetRank = std::ceil(quantile * count_);
    uint64_t prefix = 0;
    int foundKey = 0;
    // sort keys for quantile search
    vector<std::pair<int, uint64_t>> sorted;
    if (is_switched()) {
      for (auto &kv : *(map_p_)) {
        sorted.emplace_back(kv.first, kv.second);
      }
    } else {
      for (auto &kv : vec_) {
        sorted.emplace_back(kv.first, kv.second);
      }
    }
    std::sort(sorted.begin(), sorted.end(),
              [](std::pair<int, uint64_t> &a, std::pair<int, uint64_t> &b) {
                return a.first < b.first;
              });

    for (auto &kv : sorted) {
      prefix += kv.second;
      if (prefix >= targetRank) {
        foundKey = kv.first;
        break;
      }
    }
    return std::pow(1 + alpha_, std::abs(foundKey)) * ((foundKey < 0) ? -1 : 1);
  }

  uint64_t count() const { return count_; }
  double sum() const { return sum_; }
  double max() const {
    return query(1.0);
  }  // NOTE: max is an estimate value of quantile(1.0),  not precise value
  double min() const { return query(0.0); }

 private:
  double alpha_;
  size_t threshold_;

  vector<std::pair<int, uint64_t>>
      vec_;  // TODO: compress int&uint64_t to int64_t[int32_t,uint32_t]
  unique_ptr<map<int, uint64_t>> map_p_;
  uint64_t count_ = 0;
  double sum_ = 0;
};

class BucketizeEstimator : public Estimator {
 public:
  // window_num: window list cache, incase of different period from est to snap
  // collector alpha: estimate precision, in double
  BucketizeEstimator(int window_num = 10, double alpha = Bucket_kAlpha)
      : alpha_(alpha) {
    windows_.reserve(window_num);
    window_pos_ = 0;
    for (int i = 0; i < window_num; ++i) {
      windows_.emplace_back(alpha_);
    }
    windows_[window_pos_].reset();
  }
  void observe(double value) override { windows_[window_pos_].insert(value); }
  double query(double quantile) override {
    return windows_[window_pos_].query(quantile);
    // TODO: Multi window_ merge method, neednt for now
    //  SimpleBucket merged(alpha_);
    //    for (auto &w : windows_) merged.merge(w);
    //  return merged.query(quantile);
  }
  void reset() override {
    size_t new_window_pos_ = (window_pos_ + 1) % windows_.size();
    windows_[new_window_pos_].reset();
    window_pos_ = new_window_pos_;
  }
  uint64_t count() const override { return windows_[window_pos_].count(); }
  double sum() const override { return windows_[window_pos_].sum(); }
  double max() const override { return windows_[window_pos_].max(); }
  double min() const override { return windows_[window_pos_].min(); }
  double last() const override {
    throw std::runtime_error("BucketizeEstimator::last() not implemented");
  }

 private:
  double alpha_;
  size_t window_pos_;
  vector<SimpleBucket> windows_;
};

}  // namespace monitor
}  // namespace recis

#endif