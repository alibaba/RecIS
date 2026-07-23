#ifndef PAIIO_THIRD_PARTY_COMMOM_UTIL_H_
#define PAIIO_THIRD_PARTY_COMMOM_UTIL_H_

#include <vector>
#include <string>
#include <memory>
#include <queue>
#include <thread>
#include <future>
#include <condition_variable>
#include <functional>
#include <iostream>

namespace xdl
{
namespace paiio
{
namespace third_party
{
namespace common_util
{

std::vector<std::string> StrSplit(const std::string& text, 
    const std::string& sepStr, bool ignoreEmpty);
std::string strip(const std::string& str);
std::vector<std::string> FilterEmptyStr(const std::vector<std::string>& origin_str_vec);
std::string JoinStr(const std::string& str_vec, const std::string& sep);

class SimpleThreadPool {
  public:
    static SimpleThreadPool* GetInstance() {
      static std::unique_ptr<SimpleThreadPool> thread_pool(new SimpleThreadPool(16));
      return thread_pool.get();
    }

    SimpleThreadPool(SimpleThreadPool& ) = delete;
    SimpleThreadPool& operator=(SimpleThreadPool& ) = delete;

    template<typename FuncType, typename... Args>
    auto Enqueue(FuncType&& Func, Args&&... args)
      -> std::future<typename std::result_of<FuncType(Args...)>::type> {
      using return_type = typename std::result_of<FuncType(Args...)>::type;
      auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<FuncType>(Func), std::forward<Args>(args)...)
          );
      std::future<return_type> ret = task->get_future();
      {
        std::lock_guard<std::mutex> lock(task_queue_lock_);
        task_queue_.push([task] () {(*task)();});
      }
      task_queue_cond_.notify_all();
      return ret;
    }

  ~SimpleThreadPool() {
    {
    std::lock_guard<std::mutex> lock(task_queue_lock_);
    stop_ = true;
    }
    task_queue_cond_.notify_all();
    for(auto & worker: workers_) {
      worker.join();
    }
  }
  private:

    SimpleThreadPool(int thread_cnt) {
      thread_cnt_ = thread_cnt;
      stop_ = false;
      for (int i = 0; i < thread_cnt; i++) {
        workers_.emplace_back(
            [this] () {
            this->WorkLoop();
            });
      } 
    }

    void WorkLoop() {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->task_queue_lock_);
          task_queue_cond_.wait(lock, 
              [this] () {return !this->task_queue_.empty()||stop_;});
          if (this->stop_) {
            return;
          }
          task = std::move(this->task_queue_.front());
          this->task_queue_.pop();
        }
        task();
      }
    }
    

    int thread_cnt_;
    bool stop_;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex task_queue_lock_;
    std::condition_variable task_queue_cond_;
};

} // common_util
} // third_party
} // paiio
} // xdl
#endif // PAIIO_THIRD_PARTY_COMMOM_UTIL_H_
