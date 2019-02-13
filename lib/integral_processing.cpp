#include "integral_processing.hpp"

#include <iostream>
#include "boost/asio.hpp"
#include "boost/thread/thread.hpp"

namespace sp {

integral_computation::integral_computation(unsigned int thread_count) {
    const auto possible_threads = boost::thread::hardware_concurrency();
    if(thread_count == 0 || thread_count > possible_threads)
        thread_count = possible_threads;
    workers = std::make_unique<thread_pool_t>(thread_count);
}

bool integral_computation::enqueue_file(const std::string& id, cv::Mat& mat) {
    for(auto i = 0; i < mat.channels(); ++i) {
        auto task = std::make_unique<processing_context>(id, mat, i);
        boost::asio::post(*workers, [this, task = std::move(task)]() mutable {
            try {
                task->execute();
            }
            catch(const std::exception& exception) {
                std::cerr << task->get_id() << ": " << exception.what();
            }
            boost::asio::post(io, [this, task = std::move(task)]() mutable {
                on_complete(std::move(task));
            });
        });
    }

    return true;
}

void integral_computation::set_on_complete(on_complete_fn_t fn) {
    on_complete_fn = fn;
}

void integral_computation::wait_for_complete() {
    workers->join();
    io.run();
}

void sp::integral_computation::on_complete(processing_context::ptr task) {
    const auto filename = task->get_id();
    const std::size_t channels = task->get_image().channels();
    std::cout << "Task completed: " << task->get_id() << " ["
              << task->get_channel() + 1 << "/" << channels << "]" << std::endl;

    auto& tasks = completed_tasks[filename];
    tasks.emplace_back(std::move(task));
    if(tasks.size() < channels)
        return;

    std::sort(
        std::begin(tasks), std::end(tasks),
        [](const auto& task1, const auto& task2) {
            return task1->get_channel() < task2->get_channel();
        });

    if(on_complete_fn)
        on_complete_fn(tasks);
    completed_tasks.erase(filename);
}

} // namespace sp