///
/// \file
/// Defines the sp::integral_processing class, which uses as computation
/// executor
///

#pragma once
#include "boost/asio/io_context.hpp"
#include "boost/asio/thread_pool.hpp"
#include "opencv2/imgproc.hpp"
#include "processing_context.hpp"

namespace sp {

/** @brief Incapsulates execution logic for integal image computation.

Serves matrix decomposition to several tasks, which will be computed
asynchronously in separated thread(s). Thread pool may be
extended to number of CPU cores.
After computation collects tasks of the same input matrix to vector and call
completion handler for that vector.

Usage example:
@code
    integral_computation executor;
    executor.set_on_complete([&](const auto& tasks) {
        for(const auto& task : tasks)
            std::cout << "task result: " << task->get_result();
    });
    const auto file_name = "Lena.png";
    cv::Mat some_mat = cv::imread(file_name);
    executor.enqueue_file(file_name, some_mat);
    executor.wait_for_complete();
@endcode
*/
class integral_computation {
public:
    //! @brief Alias to vector of completed tasks
    using task_set_t = std::vector<processing_context::ptr>;

    //! @brief Shorthand for completion handler signature
    using on_complete_fn_t = std::function<void(const task_set_t& tasks)>;

private:
    //! @brief Shorthand for thread pool class.
    using thread_pool_t = boost::asio::thread_pool;

    //! @brief Shorthand to grouping tasks of the same matrix according its id.
    using file_tasks_t = std::map<std::string, task_set_t>;

public:
    /** @brief Constructs computation executer

    Executor contains several threads and performs computation for each channel
    in different threads.
    @param thread_count Number of threads for computation.
    @note In case if thread_count == 0 number of threads will be the same as
    the CPU core numbers
    */
    integral_computation(unsigned int thread_count = 0);

    /** @brief Enqueues matrix with its corresponding identification

    @param id Matrix identification.
    @param mat Matrix.
    */
    bool enqueue_file(const std::string& id, cv::Mat& mat);

    /** @brief Sets completion handler.
    
    Completion handler will be invoked after the matrix processed.
    @param fn Completion handler.
    */
    void set_on_complete(on_complete_fn_t fn);

    /** @brief Waiting for completion of the matrix calculation.

    Blocks until all matrices are calculated and call completition handler for
    each vector.
    */
    void wait_for_complete();

private:
    //! @brief Stores completed task to corresponded vector
    void on_complete(processing_context::ptr task);

private:
    boost::asio::io_context io;
    std::unique_ptr<thread_pool_t> workers;
    file_tasks_t completed_tasks;
    on_complete_fn_t on_complete_fn;
};

} // namespace sp