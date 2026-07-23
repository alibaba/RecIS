#ifndef PAIIO_METRICS_OPENSTORAGE_METRIC_REPORTER_H_
#define PAIIO_METRICS_OPENSTORAGE_METRIC_REPORTER_H_
#pragma once

#include <memory>
#include <vector>
#include <utility>
#include <chrono>
#include <mutex>
#include <stdio.h>
#include <unistd.h>
// #include <condition_variable>
// #include <regex>
// #include <rapidjson/document.h>
// #include "kmonitor/client/common/Common.h"
// #include "kmonitor/client/core/MetricsTags.h"
// #include "kmonitor/client/core/MetricsConfig.h"
// #include "kmonitor/client/core/MutableMetric.h"
// #include "kmonitor/client/StatisticsType.h"
// #include "kmonitor/client/KMonitor.h"
// #include "kmonitor/client/KMonitorFactory.h"

#include "column-io/monitor/metric_client.h"


namespace openstorage {
//  MetricReporter = client + client_tag
// typedef MetricReporter = std::pair<std::shared_ptr<recis::monitor::Client>, std::shared_ptr<recis::monitor::PointTag>>;
// TODO: make global variables weak so that multi include unit canbe link
// static std::atomic<int> factory_client_cnt_(0); // 初始化的client数目. 全部销毁后自动执行Factory Shutdown
// static std::string factory_alog_config_base = "\n"
//                         "inherit.kmonitor=false\n"
//                         "alog.appender.KmonAppender=FileAppender\n"
//                         "alog.appender.KmonAppender.fileName=VAR_STD_LOG_DIR/kmonitor.log\n"
//                         "alog.appender.KmonAppender.flush=false\n"
//                         "alog.appender.KmonAppender.cache_limit=128\n"
//                         "alog.appender.KmonAppender.max_file_size=100\n"
//                         "alog.appender.KmonAppender.compress=true\n"
//                         "alog.appender.KmonAppender.log_keep_count=5\n"
//                         "alog.appender.KmonAppender.layout=PatternLayout\n"
//                         "alog.appender.KmonAppender.layout.LogPattern=[%%d] [%%l] [%%F:%%n] [%%m]\n"
//                         "inherit.kmonmetrics=false\n"
//                         "alog.appender.KmonMetricsAppender=FileAppender\n"
//                         "alog.appender.KmonMetricsAppender.fileName=VAR_STD_LOG_DIR/kmonitor_metrics.log\n"
//                         "alog.appender.KmonMetricsAppender.flush=false\n"
//                         "alog.appender.KmonMetricsAppender.cache_limit=32\n"
//                         "alog.appender.KmonMetricsAppender.max_file_size=100\n"
//                         "alog.appender.KmonMetricsAppender.compress=true\n"
//                         "alog.appender.KmonMetricsAppender.log_keep_count=5\n"
//                         "alog.appender.KmonMetricsAppender.layout=PatternLayout\n"
//                         "alog.appender.KmonMetricsAppender.layout.LogPattern=%%m\n";
//
// static std::string namespace_default = "xdl.metrics"; // KMonitor的namespace, 作用于全部metric_name的头部
// Metric状态枚举定义
enum MetricStatus {
    // expected status
    SUCCESS = 0, // ok
    WAITING = 1001, // waiting status
    
    // cross-processes  errors
    REQUEST_ERROR = 2001, // http, rpc request error
    CALL_ERROR = 2002, //ipc, internal-func call error

    // logic error in process
    JSON_ERROR = 3001, //
    CODEC_ERROR = 3002, // e.g. utf-8 error
    FIELD_ERROR = 3003, // field missing, type-wrong or other relative error
    OUTDATE_ERROR = 3011,  // context(session, token,,,) is expired

    // un classified errors
    UNKNOWN_ERROR = 9999
};
inline const std::string GetLogLevelNameFromEnv() {
    // option: in aios/alog/src/cpp/Configurator.cpp:Configurator::getLevelByString
    //        "DISABLE"|| "FATAL"|| "ERROR"|| "WARN"|| "INFO"|| "DEBUG"|| "TRACE1"~ "TRACE3";
    const char *log_level = getenv("KMONITOR_LOG_LEVEL") ? getenv("KMONITOR_LOG_LEVEL") : getenv("NEBULA_IO_LOG_LEVEL");
    const std::vector<std::string> level_names{"DEBUG", "INFO", "WARN", "ERROR", "FATAL", "DISABLE"};
    if (log_level == nullptr) {
        // alog::Logger::getRootLogger()->setLevel(alog::LOG_LEVEL_INFO);
        return level_names[1];
    }
    return level_names[atoi(log_level)];
}
// 常用的指标名, 可以继续往下添加, 也可以直接在代码中使用字符串常量
namespace MetricName{
    const static std::string ReadBatch = "read_batch";
    const static std::string ReadBatchImpl = "read_batch_impl";
    const static std::string ReadRows = "reader_create"; // read_rows语义和read_batch rows容易混淆
};
// 常用的指标tag
namespace MetricTag{
    // const static kmonitor::MetricsTags Succ = kmonitor::MetricsTags(std::map<std::string, std::string>({{"code", "0"}, {"status", "success"}}));
    // const static kmonitor::MetricsTags Fail = kmonitor::MetricsTags(std::map<std::string, std::string>({{"code", "1"}, {"status", "fail"}}));
    const static std::map<std::string, std::string> Succ{{"code", "0"}, {"status", "success"}};
    const static std::map<std::string, std::string> Fail{{"code", "1"}, {"status", "fail"}};
};

// // 最小化粒度的纵向汇聚指标线, 非UpdateKey成员元素的线在进程内汇报前会被纵向聚合(不包括global tags)
// struct UpdateKey {
//     std::string kmetric_name;
//     // const kmonitor::MetricsTags* kmetric_tag; // 若引用了 namespace MetricTag 请勿释放
//     const std::shared_ptr<recis::monitor::PointTag> kmetric_tag;
//     explicit UpdateKey(const std::string& n, const kmonitor::MetricsTags* t=&MetricTag::Succ): kmetric_name(n), kmetric_tag(t) {};
// 
//     bool operator<(const UpdateKey& other) const {
//         if (kmetric_name != other.kmetric_name) {
//             return kmetric_name < other.kmetric_name;
//         }
//         // kmonitor/client/core/MetricsTags 实现了 operator<
//         return *kmetric_tag < *(other.kmetric_tag);
//     }
// };
// struct UpdateVal{
//     std::atomic<int> count; // query_count
//     std::atomic<int> latency_ms; // query_latency_ms
//     explicit UpdateVal(int cnt = 0, int latency = 0): count(cnt), latency_ms(latency) {};
//     UpdateVal(const UpdateVal& other): count(other.count.load()), latency_ms(other.latency_ms.load()) {};
//     UpdateVal(UpdateVal&& other) noexcept: count(other.count.load()), latency_ms(other.latency_ms.load()) {}
//     UpdateVal& operator=(const UpdateVal& other) {
//         if (this == &other) return *this;
//         count.store(other.count.load());
//         latency_ms.store(other.latency_ms.load());
//         return *this;
//     }
//     // NOTE: not really moved here, this function is just for compatibility
//     UpdateVal& operator=(UpdateVal&& other) noexcept {
//         if (this == &other) return *this;
//         count.store(other.count.load());
//         latency_ms.store(other.latency_ms.load());
//         return *this;
//     }
// 
//     void Reset(){
//         count.store(0);
//         latency_ms.store(0);
//     }
//     std::pair<int, int> Get(){
//         int count_ = count.load();
//         int latency_ms_ = latency_ms.load();
//         return std::make_pair(count_, latency_ms_);
//     }
//     std::pair<int, int> GetAndReset(){
//         int count_ = count.exchange(0);
//         int latency_ms_ = latency_ms.exchange(0);
//         return std::make_pair(count_, latency_ms_);
//     }
//     void Set(int cnt, int latency = 0){
//         count.fetch_add(cnt);
//         latency_ms.fetch_add(latency);
//     }
//     void Set(UpdateVal& val_from){
//         // std::pair<int, int> val = val_from.Get();
//         count.fetch_add(val_from.count.load());
//         latency_ms.fetch_add(val_from.latency_ms.load());
//     }
// };
// 
// class MetricReporter 
// 直接用这个代替:
// std::shared_ptr<recis::monitor::client> metric_reporter_ = recis::monitor::Factory::get_client("xdl.metrics.openstorage_session_");
// 
// class MetricReporter{
// private:
//     // TODO: 这个ReporterThread是一个对象一个线程. 如果存在100个分区, 100个session reporter, 就有100个线程. 可能存在瓶颈, 但目前普通任务问题不大
//     void ReporterThread(){
//         auto target_clock = std::chrono::steady_clock::now() + std::chrono::microseconds(report_interval_);
//
//         // std::string read_batch_qps_name = client_name_ + ".read_batch_qps" + ".sum";
//         // // 加.sum后缀是因为paiio所用老版monitor_service/kmonitor-client-cpp的Gauge会强制加.sum/.count等后缀 需在grafana统一
//         // std::string read_batch_latency_name = client_name_ + ".read_batch_latency_ms";
//         // std::string read_rows_qps_name = client_name_ + ".read_rows_qps" + ".sum";
//         // std::string read_rows_latency_name = client_name_ + ".read_rows_latency_ms";
//      
//         // const kmonitor::MetricsTags succ_tag(std::map<std::string, std::string>({{"code", "0"}, {"status", "success"} }));
//         // const kmonitor::MetricsTags fail_tag(std::map<std::string, std::string>({{"code", "1"}, {"status", "fail"} }));
//         // kmonitor::MutableMetric* readbatch_count_base = kmon_client->RegisterMetric( read_batch_qps_name, kmonitor::MetricType::GAUGE, kmonitor::MetricLevel::NORMAL, common_stat_type_);
//         // kmonitor::MutableMetric* readbatch_latency_base = kmon_client->RegisterMetric(read_batch_latency_name, kmonitor::MetricType::GAUGE, kmonitor::MetricLevel::NORMAL, common_stat_type_); // StatisticsType 0 , avg
//         // kmonitor::Metric* metric_readbatch_count_succ = readbatch_count_base->DeclareMetric(&succ_tag);
//         // kmonitor::Metric* metric_readbatch_count_fail = readbatch_count_base->DeclareMetric(&fail_tag);
//         // kmonitor::Metric* metric_readbatch_latency_succ = readbatch_latency_base->DeclareMetric(&succ_tag);
//         // kmonitor::Metric* metric_readbatch_latency_fail = readbatch_latency_base->DeclareMetric(&fail_tag);
//         // kmonitor::MutableMetric* readrows_count_base = kmon_client->RegisterMetric(read_rows_qps_name, kmonitor::MetricType::GAUGE, kmonitor::MetricLevel::NORMAL, common_stat_type_);
//         // kmonitor::MutableMetric* readrows_latency_base = kmon_client->RegisterMetric(read_rows_latency_name, kmonitor::MetricType::GAUGE, kmonitor::MetricLevel::NORMAL, common_stat_type_); // StatisticsType 0 , avg
//         // kmonitor::Metric* metric_readrows_count_succ = readrows_count_base->DeclareMetric(&succ_tag);
//         // kmonitor::Metric* metric_readrows_count_fail = readrows_count_base->DeclareMetric(&fail_tag);
//         // kmonitor::Metric* metric_readrows_latency_succ = readrows_latency_base->DeclareMetric(&succ_tag);
//         // kmonitor::Metric* metric_readrows_latency_fail = readrows_latency_base->DeclareMetric(&fail_tag);
//         // 由下个上报周期强制回收该metric: Metric* metric; metric->Release();
//         // 含义:  query接口+query tag -->  ( count Metric*, latency_ms Metric*)
//         std::map<UpdateKey, std::pair<kmonitor::Metric*,kmonitor::Metric*>> metrics_list;
//
//         while(running_){ // E.g. xdl.metrics.openstorage_session_cache.cache_qps
//             {
//                 std::lock_guard<std::mutex> lock(kmetric_query_mutex_);
//                 for(auto& query : kmetric_query_){
//                     const UpdateKey& key = query.first;
//                     std::pair<int,int> value_pair = query.second.GetAndReset();
//
//                     std::string query_count_name = client_name_ + "." + key.kmetric_name + "_" + "qps.sum";//TODO: update kmonitor and add postfix .sum and api statisticsType
//                     std::string query_latency_name = client_name_ + "." + key.kmetric_name + "_" + "latency_ms"; 
//                  
//                     if (metrics_list.find(key) == metrics_list.end()){
//                         const kmonitor::MetricsTags* tag_ptr = key.kmetric_tag;
//                         kmonitor::MutableMetric* query_count_base = kmon_client->RegisterMetric(query_count_name, kmonitor::MetricType::GAUGE, kmonitor::MetricLevel::NORMAL, common_stat_type_);
//                         kmonitor::MutableMetric* query_latency_base = kmon_client->RegisterMetric(query_latency_name, kmonitor::MetricType::GAUGE, kmonitor::MetricLevel::NORMAL, common_stat_type_);
//
//                         kmonitor::Metric* query_count_const = query_count_base->DeclareMetric(tag_ptr);
//                         kmonitor::Metric* query_latency_const = query_latency_base->DeclareMetric(tag_ptr);
//                         metrics_list[key] = std::make_pair(query_count_const, query_latency_const);
//                     }
//                     auto const_metric_pair = metrics_list[key];
//                     kmonitor::Metric* query_count_const = const_metric_pair.first;
//                     kmonitor::Metric* query_latency_const = const_metric_pair.second;
//
//                     kmon_client->Report(query_count_const, value_pair.first);
//                     kmon_client->Report(query_latency_const, value_pair.second);
//                 }
//             }
//             std::unique_lock<std::mutex> lock(thread_mtx);
//             while (running_ && std::chrono::steady_clock::now() < target_clock) {
//                 thread_cv.wait_until(lock, target_clock, [&]{ return !running_; });
//             }
//             target_clock += std::chrono::microseconds(report_interval_);
//         }
//         for(auto it : metrics_list) {
//             auto metric_pair = it.second;
//             metric_pair.first->Release();
//             metric_pair.second->Release();
//         }
//         return;
//     }
//
    /* 
     * @brief 获取自身被父进程启动的顺序 (目的是擦多进程的屁股, 防止被agent误聚合)
     * @return int 启动顺序
     * 进程间通过不同slice id实现多读, 但metric汇聚时 通过slice id或进程号等方式区分数据源 其开销过大.
     * 因此这里通过get_child_order获取共享父进程的不同IO子进程的启动顺序, 并在metric tag中添加该顺序而区分出来
     */
    static int get_child_order_internal() {
        pid_t my_pid = getpid();
        pid_t parent_pid = getppid();
        char path[256];
        snprintf(path, sizeof(path), "/proc/%d/task/%d/children", parent_pid, parent_pid);
        
        FILE* f = fopen(path, "r");
        if (!f) return -1; // 打开错误, 返回默认值
        int count = 0;
        pid_t child_pid;
        while (fscanf(f, "%d", &child_pid) == 1) {
            count++;
            if (child_pid == my_pid || count > 1024 ) { // 防止死循环. 虽然理论不会发生
                fclose(f);
                return count;
            }
        }
        fclose(f);
        return 0;
    }
    /**
     * @brief 获取自身被父进程启动的顺序
     * @return int 启动顺序
     */
    static int get_child_order() {
        static int order = 0;
        static std::once_flag once_flag;
        std::call_once(once_flag, []() {
            order = get_child_order_internal();
        });
        return order;
    }
//
//     static std::map<std::string, std::string> tags_from_env(){ // 从env获取常见tags
//         static std::map<std::string, std::string> tags;
//         static std::once_flag tags_init_flag_;
//         std::call_once(tags_init_flag_, []{
//             // 获取基本信息
//             // SIGMA_APP_SITE
//             tags["sigma_app_site"] = getenv("SIGMA_APP_SITE") ? getenv("SIGMA_APP_SITE") : "null";
//             tags["calculate_cluster"] = getenv("CALCULATE_CLUSTER") ? getenv("CALCULATE_CLUSTER") : "null";
//             tags["nebula_project"] = getenv("NEBULA_PROJECT") ? getenv("NEBULA_PROJECT") : "null";
//             tags["scheduler_queue"] = getenv("SCHEDULER_QUEUE") ? getenv("SCHEDULER_QUEUE") : "null"; // worker or scheduler or ps
//             tags["io_type"] = "column_io"; // python module name. 严格需要从pybind11获取, 图方便&少依赖 这里先硬编码
//             tags["task_id"] = getenv("TASK_ID") ? getenv("TASK_ID") : "null";
//             tags["app_id"] = getenv("APP_ID") ? getenv("APP_ID") : "null";
//             tags["user_id"] = getenv("_NEBULA_USER_ID") ? getenv("_NEBULA_USER_ID") : "null";
//             tags["task_name"] = getenv("TASK_NAME") ? getenv("TASK_NAME") : "worker"; // worker or scheduler or ps
//             tags["openstorage_backend"] = getenv("OPEN_STORAGE_BACKEND") ? getenv("OPEN_STORAGE_BACKEND") : "null";
//             tags["tunnel_endpoint"] = getenv("OPEN_STORAGE_TUNNEL_ENDPOINT") ? getenv("OPEN_STORAGE_TUNNEL_ENDPOINT") : "null";
//             tags["host_ip"] = getenv("KMONITOR_SINK_ADDRESS") ? getenv("KMONITOR_SINK_ADDRESS") : "localhost";
//             // 获取"docker_image"配置, 只获取镜像tag, 不包含repo name
//             std::string docker_image_ = getenv("docker_image")  ? getenv("docker_image")            : "null";
//             docker_image_ = getenv("pouch_container_image")     ? getenv("pouch_container_image")   : docker_image_;
//             size_t name_index_ = docker_image_.find_last_of(":");
//             tags["docker_image"] = (name_index_ != std::string::npos) ? docker_image_.substr(name_index_+1) : docker_image_;
//             // 获取worker task_index; 这里xdl框架风格是args parse --task_index, mdl风格为"RANK". 统一两种逻辑
//             tags["rank"] = getenv("TASK_INDEX") ? getenv("TASK_INDEX")  : "0";
//             tags["rank"] = getenv("RANK")       ? getenv("RANK")        : tags["rank"];
//             tags["multiprocess_seq"] = std::to_string(get_child_order()); // MultiDataset间的子进程顺序
//         });
//         return tags;
//     }
//
// public:
//     MetricReporter(const std::string& client_name="openstorage_default", const std::map<std::string, std::string>& tags={}, const std::string& sink_address="") : \
//         namespace_(namespace_default), client_name_(client_name), running_(true) {
//         // read_batch_count_succ_ = 0;
//         // read_batch_count_fail_ = 0;
//         // read_batch_latency_succ_ = 0;
//         // read_batch_latency_fail_ = 0;
//         // read_rows_count_succ_ = 0;
//         // read_rows_count_fail_ = 0;
//         // read_rows_latency_succ_ = 0;
//         // read_rows_latency_fail_ = 0;
//         {
//             std::lock_guard<std::mutex> lock(kmetric_query_mutex_);
//             kmetric_query_.clear();
//         }
//         global_tags_ = MetricReporter::tags_from_env(); // std::cerr << "[INFO] global_tags_ is empty, fill it with job info" << std::endl;
//         for(auto it = tags.begin(); it != tags.end(); ++it) {
//             global_tags_[it->first] = it->second;
//         }
//
//         if( !kmonitor::KMonitorFactory::IsStarted() ){
//             std::string level = GetLogLevelNameFromEnv();
//             std::string log_dir = (getenv("STD_LOG_DIR") != nullptr) ? getenv("STD_LOG_DIR")    :   "/var/log/";
//             std::string full_alog_config = "alog.logger.kmonitor=" + level + ", KmonAppender\n" "alog.logger.kmonmetrics=" + level +", KmonMetricsAppender\n" + factory_alog_config_base;
//             full_alog_config = std::regex_replace(full_alog_config, std::regex("VAR_STD_LOG_DIR"), log_dir);
//             AUTIL_LOG_CONFIG_FROM_STRING(full_alog_config.c_str()); // AUTIL_ROOT_LOG_CONFIG();
//             kmonitor::MetricsConfig metrics_config;
//             metrics_config.set_inited(true);
//             metrics_config.set_service_name(namespace_default);
//             metrics_config.set_tenant_name("default");
//             std::string sink_address_ip = getenv("KMONITOR_SINK_ADDRESS") ? getenv("KMONITOR_SINK_ADDRESS") : "localhost";
//             // std::string sink_address_ip = getenv("KUBERNETES_NODE_IP") ? getenv("KUBERNETES_NODE_IP") : "localhost";
//             if( sink_address_ip.find(":") != sink_address_ip.npos )
//                 sink_address_ip = "[" + sink_address_ip + "]";
//             metrics_config.set_sink_address(sink_address_ip + ":4141"); // default: "localhost:4141"
//             auto global_tags_ = MetricReporter::tags_from_env();
//             for (auto it = global_tags_.begin(); it != global_tags_.end(); ++it) {
//                 metrics_config.AddGlobalTag(it->first, it->second); // metrics_config.AddCommonTag(it->first, it->second); 
//             }
//             std::cerr << "[INFO] start init kmonitor::KMonitorFactory BY AUTIL_LOG_CONFIG_FROM_STRING:"<< level << ", service_name:" << namespace_default << ", sink addr:" << metrics_config.sink_address() << ", global map count:" << global_tags_.size() << std::endl;
//             if ( !kmonitor::KMonitorFactory::Init(metrics_config)) {
//                 std::cerr << "[ERROR] INIT kmonitor::KMonitorFactory FAIL" << std::endl;
//                 throw std::invalid_argument("init kmonitor factory failed");
//             }
//             kmonitor::KMonitorFactory::Start();
//         }
//         factory_client_cnt_.fetch_add(1);
//         // std::cout << "[DEBUG] create new kmonitor::KMonitor client" << std::endl;
//         common_stat_type_ |= kmonitor::StatisticsType::MIN_MAX;
//         common_stat_type_ |= kmonitor::StatisticsType::SUMMARY;
//         common_stat_type_ |= kmonitor::StatisticsType::PERCENTILE_50;
//         common_stat_type_ |= kmonitor::StatisticsType::PERCENTILE_99;
//         kmon_client = kmonitor::KMonitorFactory::GetKMonitor(client_name_);
//         reporter_thread_ = std::thread(&MetricReporter::ReporterThread, this);
//         // TODO: (sess|proj|table|part) tag需要(由thread)动态创建MutableMetric, 暂时无法由构造函数事先确定, 先固定唯一
//     }
//     ~MetricReporter(){
//         {
//             std::lock_guard<std::mutex> lock(thread_mtx);
//             running_ = false;
//         }
//         if (reporter_thread_.joinable()){
//             reporter_thread_.join();
//         }
//         // If all processes exit (no other session, except for me), then shutdown kmonitor
//         // std::cout << "[DEBUG] begin kmonitor::KMonitorFactory::Shutdown in ~MetricReporter factory_client_cnt_:" << factory_client_cnt_.load() << " , IsStarted:" << kmonitor::KMonitorFactory::IsStarted() << std::endl;
//         // if( factory_client_cnt_.fetch_sub(1) == 1  ){
//         //     kmonitor::KMonitorFactory::Shutdown();
//         // }
//         // std::cout << "[DEBUG] finish kmonitor::KMonitorFactory::Shutdown in ~MetricReporter" << std::endl;
//     }
//     // void UpdateReadBatch(int count=1, int latency_ms=0, bool succ = true){
//     //     // 统计read_batch状态, 汇报由(ReporterThread)reporter_thread_负责
//     //     if(succ ){
//     //         read_batch_count_succ_.fetch_add(count);
//     //         read_batch_latency_succ_.fetch_add(latency_ms);
//     //     }else{
//     //         read_batch_count_fail_.fetch_add(count);
//     //         read_batch_latency_fail_.fetch_add(latency_ms);
//     //     }
//     //     // std::cout << "[DEBUG] UpdateReadBatch count:"<< read_batch_count_succ_.load() << "read_batch_latency_succ_:"<< read_batch_latency_succ_.load() << ", read_batch_latency_fail_:" << read_batch_latency_fail_.load() << " [DEBUG] " << std::endl;
//     // }
//     // void UpdateReadRows(int count=1, int latency_ms=0, bool succ = true){
//     //     // 统计read_rows状态, 汇报由(ReporterThread)reporter_thread_负责
//     //     if(succ ){
//     //         read_rows_count_succ_.fetch_add(count);
//     //         read_rows_latency_succ_.fetch_add(latency_ms);
//     //     }else{
//     //         read_rows_count_fail_.fetch_add(count);
//     //         read_rows_latency_fail_.fetch_add(latency_ms);
//     //     }
//     //     // std::cout << "[DEBUG] UpdateReadRows count:"<< read_rows_count_succ_.load() << "read_rows_latency_succ_:"<< read_rows_latency_succ_.load() << ", read_rows_latency_fail_:" << read_rows_latency_fail_.load() << " [DEBUG] " << std::endl;
//     // }
//
//     void UpdateQuery(UpdateKey& key, UpdateVal& value){
//         std::lock_guard<std::mutex> lock(kmetric_query_mutex_);
//         if(kmetric_query_.find(key) == kmetric_query_.end()){
//             kmetric_query_[key] = UpdateVal();
// 
//         }
//         kmetric_query_[key].Set(value);
//     }
//
//     static void UpdateImmediate(std::string name, int val, const std::map<std::string, std::string>& tags={}, std::string client_name="openstorage_default"){
//        // 直接汇报, 不经过独立线程的统计和汇总
//        // if (factory_client_cnt_.fetch_sub(1) <= 1 && kmonitor::KMonitorFactory::IsStarted() ){
//        if( !kmonitor::KMonitorFactory::IsStarted() ){
//            std::string level = GetLogLevelNameFromEnv();
//            std::string log_dir = (getenv("STD_LOG_DIR") != nullptr) ? getenv("STD_LOG_DIR")    :   "/var/log/";
//            std::string full_alog_config = "alog.logger.kmonitor=" + level + ", KmonAppender\n" "alog.logger.kmonmetrics=" + level +", KmonMetricsAppender\n" + factory_alog_config_base;
//            full_alog_config = std::regex_replace(full_alog_config, std::regex("VAR_STD_LOG_DIR"), log_dir);
//            AUTIL_LOG_CONFIG_FROM_STRING(full_alog_config.c_str()); // AUTIL_ROOT_LOG_CONFIG();
//            kmonitor::MetricsConfig metrics_config;
//            metrics_config.set_inited(true);
//            metrics_config.set_service_name(namespace_default);
//            metrics_config.set_tenant_name("default");
//            std::string sink_address_ip = getenv("KMONITOR_SINK_ADDRESS") ? getenv("KUBERNETES_NODE_IP") : "localhost";
//            // std::string sink_address_ip = getenv("KUBERNETES_NODE_IP") ? getenv("KUBERNETES_NODE_IP") : "localhost";
//            if( sink_address_ip.find(":") != sink_address_ip.npos )
//                sink_address_ip = "[" + sink_address_ip + "]";
//            metrics_config.set_sink_address(sink_address_ip + ":4141"); // default: "localhost:4141"
//            auto global_tags_ = MetricReporter::tags_from_env();
//            for (auto it = global_tags_.begin(); it != global_tags_.end(); ++it) {
//                metrics_config.AddGlobalTag(it->first, it->second); // metrics_config.AddCommonTag(it->first, it->second); 
//            }
//            std::cerr << "[INFO] start init kmonitor::KMonitorFactory BY AUTIL_LOG_CONFIG_FROM_STRING:"<< level << ", service_name:" << namespace_default << ", sink addr:" << metrics_config.sink_address() << ", global map count:" << global_tags_.size() << std::endl;
//            if( !kmonitor::KMonitorFactory::Init(metrics_config)) {
//                std::cerr << "[ERROR] INIT kmonitor::KMonitorFactory FAIL" << std::endl;
//                throw std::invalid_argument("init kmonitor factory failed");
//            }
//            kmonitor::KMonitorFactory::Start();
//        }
//        factory_client_cnt_.fetch_add(1);
//        auto tmp_client = kmonitor::KMonitorFactory::GetKMonitor(client_name);
//        auto tmp_client = recis::monitor::Factory::StaticInstance()->get_client(client_name);
//        name = client_name + "." + name; // (namespace).(client_name).(name) e.g. xdl.metrics.openstorage_reader.read_batch_qps
//         std::shared_ptr<recis::monitor::Client> tmp_client = recis::monitor::Factory::StaticInstance()->get_client(namespace_ + client_name);
//        tmp_client->RegisterMetric(name, kmonitor::MetricType::GAUGE, kmonitor::MetricLevel::NORMAL, kmonitor::StatisticsType::DEFAULT);
//         std::shared_ptr<recis::monitor::PointTag> tag = std::make_shared<recis::monitor::PointTag>(tags);
//         tmp_client->report(name, val, tag);
//        if (!tags.empty()) {
//            kmonitor::MetricsTags tags_;
//            for(auto it = tags.begin(); it != tags.end(); ++it) {
//                tags_.AddTag(it->first, it->second);
//            }
//            tmp_client->Report(name, &tags_, val);
//        }else{
//            tmp_client->Report(name, val);
//        }
//        If global process exit (no session exist except for closing me), then shutdown kmonitor
//        std::cout << "[DEBUG] begin kmonitor::KMonitorFactory::Shutdown in UpdateImmediate factory_client_cnt_:" << factory_client_cnt_.load() << " , IsStarted:" << kmonitor::KMonitorFactory::IsStarted() << std::endl;
//        if( factory_client_cnt_.fetch_sub(1) == 1  ){
//            kmonitor::KMonitorFactory::Shutdown();
//        }
//         std::cout << "[DEBUG] finish kmonitor::KMonitorFactory::Shutdown in UpdateImmediate"  << std::endl;
//     }
//
// private:
//     std::string namespace_; // xdl.metrics
//     std::string client_name_; // openstorage_reader
//     std::map<std::string, std::string> global_tags_;
//     std::map<UpdateKey, UpdateVal> kmetric_query_;
//     std::mutex kmetric_query_mutex_;
//
//     // name = ***    
//
//     bool running_;
//     std::mutex thread_mtx;
//     std::condition_variable thread_cv;
//     // kmonitor::KMonitor* kmon_client;
//     std::shared_ptr<recis::monitor::Client> kmon_client;
//     uint64_t report_interval_ = (uint64_t)kmonitor::MetricLevel::NORMAL * 1000 * 1000; // micro seconds
//     std::thread reporter_thread_;
//     int common_stat_type_ = 0;
// };


struct DeferUpdater {
    DeferUpdater(std::pair<std::shared_ptr<recis::monitor::Client>, std::shared_ptr<recis::monitor::PointTag>> metric_reporter,
                const std::string& name, int64_t v_count = 1) :metric_reporter_(metric_reporter), name_(name), val_count_(v_count) {
        if (!metric_reporter_.first || !metric_reporter_.second) {
            throw std::invalid_argument("invalid argument null Client||PointTag for Metric DeferUpdater");
        }
        if (name_.empty()) {
            throw std::invalid_argument("invalid argument null report name for Metric DeferUpdater");
        }
    }
    ~DeferUpdater() {
        auto end_time = std::chrono::steady_clock::now();
        int64_t elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        auto pTag = recis::monitor::PointTag::Clone(metric_reporter_.second, success_? MetricTag::Succ : MetricTag::Fail);
        metric_reporter_.first->report(name_ + "_qps", val_count_, pTag, recis::monitor::PointType::kGauge);
        metric_reporter_.first->report(name_ + "_latency_ms", elapsed_time_ms, pTag, recis::monitor::PointType::kGauge);
        // int64_t current_count = val_->count.load();
        // val_->Set(current_count, elapsed_time_ms);
        // auto reporter = reporter_.lock();
        // if (!reporter) {
        //     return;
        // }
        // 调用 MetricReporter 来更新指标
        // reporter->UpdateQuery(*key_, *val_);
    }
    // std::shared_ptr<UpdateKey> Key(){ return key_; }
    // std::shared_ptr<UpdateVal> Val(){ return val_; }
    int GetElapsedTimeMs(){
        auto now_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_time).count();
    }
    void SetSuccess(bool success){
        success_ = success;
    }
    // TOOD 实现MakeSureUpdate(), 析构时UpdateQuery后强制sleep interval确保上报
private:
    // std::shared_ptr<recis::monitor::Client> metric_cli_;// reporter对象一般被外部持有, 请不要在这里释放
    std::pair<std::shared_ptr<recis::monitor::Client>, std::shared_ptr<recis::monitor::PointTag>> metric_reporter_;
    std::string name_;
    // std::shared_ptr<UpdateVal> val_;
    int64_t val_count_ = 1;
    bool success_ = true;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

}  // namespace openstorage

#endif  // PAIIO_METRICS_OPENSTORAGE_METRIC_REPORTER_H_
