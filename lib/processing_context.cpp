#include "processing_context.hpp"

#include <sstream>

namespace sp {
namespace {

template<typename T>
void compute(const cv::Mat& src, cv::Mat& dst, int channel) {
    std::vector<double> last_row(src.cols, 0.0);

    const auto channels = src.channels();
    for(auto i = 0; i < src.rows; ++i) {
        double row_sum = 0.0;
        const auto src_row = src.ptr<T>(i);
        const auto dst_row = dst.ptr<double>(i);
        for(auto j = 0; j < src.cols; ++j) {
            const auto col = j * channels + channel;
            const auto value = src_row[col];
            row_sum += value + last_row[j];

            last_row[j] = value + last_row[j];
            dst_row[j] = row_sum;
        }
    }
}

} // namespace

const std::string& processing_context::get_id() const noexcept {
    return id;
}

int processing_context::get_channel() const noexcept {
    return channel;
}

const cv::Mat& processing_context::get_result() const noexcept {
    return result;
}

const cv::Mat& processing_context::get_image() const noexcept {
    return image;
}

void processing_context::execute() {
    assert(!id.empty());

    switch(image.depth()) {
    case CV_8U:
        return compute<std::uint8_t>(image, result, channel);
    case CV_16U:
        return compute<std::uint16_t>(image, result, channel);
    case CV_16S:
        return compute<std::int16_t>(image, result, channel);
    default: {
        std::stringstream error;
        error << "Unsupported image depth: " << image.depth() << ";";
        throw std::runtime_error(error.str());
    }
    }
}

} // namespace sp