#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "boost/thread/thread.hpp"
#include "integral_processing.hpp"
#include "opencv2/imgcodecs.hpp"

namespace fs = boost::filesystem;
namespace opt = boost::program_options;

namespace {

int thread_count;
std::vector<std::string> files;

void validate_params(int argc, char* argv[]) {
    opt::options_description desc;
    desc.add_options()(
        ",i", opt::value<std::vector<std::string>>(), "list of input files")(
        ",t", opt::value<int>()->default_value(0),
        "specify processing thread numbers");
    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    if(vm.count("-i")) {
        const auto& opt_files = vm["-i"].as<std::vector<std::string>>();
        std::copy(
            std::begin(opt_files), std::end(opt_files),
            std::back_inserter(files));
    }

    thread_count = vm["-t"].as<int>();
    const auto allow_threads = boost::thread::hardware_concurrency();
    if(thread_count >= 0 && thread_count < static_cast<int>(allow_threads))
        return;

    std::stringstream error;
    error << "unsupported thred number specified: " << thread_count
          << "; allowed thread number: " << allow_threads;
    throw std::out_of_range(error.str());
}

void write_on_disk(const sp::integral_computation::task_set_t& tasks) {
    const auto id = tasks.front()->get_id();
    const auto dot_index = id.find_last_of(".");
    const auto dst = std::string(id.substr(0, dot_index)).append(".integral");
    const auto filename = fs::path(dst).filename().string();

    std::cout << filename << ": merging..." << std::endl;
    std::ofstream output(dst);
    output << std::fixed << std::setprecision(1);

    auto task_count = tasks.size();
    for(const auto& task : tasks) {
        const auto& result = task->get_result();
        for(auto i = 0; i < result.rows; ++i) {
            const auto row = result.ptr<double>(i);
            for(auto j = 0; j < result.cols; ++j) {
                const auto value = row[j];
                output << value;
                if(j != result.cols - 1)
                    output << " ";
            }
            output << "\n";
        }
        if(--task_count)
            output << "\n";
    }

    output.flush();
    std::cout << filename << ": merge complete" << std::endl;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        validate_params(argc, argv);
    }
    catch(const std::exception& exception) {
        std::cerr << exception.what();
        return EXIT_FAILURE;
    }

    sp::integral_computation executor(thread_count);
    executor.set_on_complete(write_on_disk);
    std::for_each(
        std::begin(files), std::end(files), [&executor](const auto& file) {
            auto image = cv::imread(file);
            if(image.empty()) {
                std::cerr << "Unable to process: " << file << std::endl;
                return;
            }
            executor.enqueue_file(file, image);
        });

    executor.wait_for_complete();

    return 0;
}