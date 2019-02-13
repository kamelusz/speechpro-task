#include "common.hpp"

#include <iostream>

#include "boost/filesystem.hpp"
#include "integral_processing.hpp"
#include "opencv2/imgproc.hpp"

namespace fs = boost::filesystem;

namespace sp {
namespace test {
namespace {

static std::map<std::string, int> depth_to_id = {
    std::make_pair("CV_8U", CV_8U), // std::uint8_t
    std::make_pair("CV_8S", CV_8S), // std::int8_t
    std::make_pair("CV_16U", CV_16U), // std::uint16_t
    std::make_pair("CV_16S", CV_16S), // std::int8_t
    std::make_pair("CV_32S", CV_32S), // std::int32_t
    std::make_pair("CV_32F", CV_32F), // float
    std::make_pair("CV_64F", CV_64F), // double
};

template<typename T>
std::enable_if_t<sizeof(T) != 1> read_value(
    std::istringstream& line, T& value) {
    line >> value;
}

template<typename T>
std::enable_if_t<sizeof(T) == 1> read_value(
    std::istringstream& line, T& value) {
    int tmp = 0;
    if(line >> tmp)
        value = static_cast<T>(tmp);
}

template<typename T>
void read_row(
    std::istringstream& line, cv::Mat& mat, int row, int channel = -1) {
    auto col = 0;
    auto data = mat.ptr<T>(row);
    auto no_channels = [&] {
        while(!line.eof())
            read_value(line, data[col++]);
    };
    auto on_channels = [&] {
        const auto channels = mat.channels();
        while(!line.eof()) {
            const auto index = col * channels + channel;
            read_value(line, data[index]);
            ++col;
        }
    };

    if(channel == -1)
        no_channels();
    else
        on_channels();
}

template<typename T>
cv::Mat parse_input_mat(std::ifstream& data, const cv::Size& size, int depth) {
    cv::Mat result(size, depth);

    // skip first line
    std::string line;
    std::getline(data, line);

    int row = 0;
    while(std::getline(data, line)) {
        std::istringstream row_stream(line);
        read_row<T>(row_stream, result, row++);
    }

    return result;
}

cv::Mat load_expect_file(const std::string& filename) {
    std::ifstream data(filename);

    std::string depth;
    int rows = 0, cols = 0, channels = 0;
    data >> depth >> rows >> cols >> channels;

    // skip first line
    std::string line;
    std::getline(data, line);

    auto row = 0;
    auto channel = 0;
    cv::Mat result(rows, cols, CV_64FC(channels));
    while(std::getline(data, line)) {
        std::istringstream row_stream(line);
        read_row<double>(row_stream, result, row, channel);
        if(++row % rows == 0) {
            std::getline(data, line);
            row = 0;
            ++channel;
        }
    }

    return result;
}

cv::Mat load_input_file(const std::string& filename) {
    std::ifstream data(filename);
    assert(data.good());

    std::string depth;
    cv::Size size;
    int channels = 0;
    data >> depth >> size.height >> size.width >> channels;

    switch(depth_to_id[depth]) {
    case CV_8U:
        return parse_input_mat<std::uint8_t>(data, size, CV_8UC(channels));
    case CV_8S:
        return parse_input_mat<std::int8_t>(data, size, CV_8SC(channels));
    case CV_16U:
        return parse_input_mat<std::uint16_t>(data, size, CV_16UC(channels));
    case CV_16S:
        return parse_input_mat<std::int16_t>(data, size, CV_16SC(channels));
    case CV_32S:
        return parse_input_mat<std::int32_t>(data, size, CV_32SC(channels));
    default:
        break;
    }

    return cv::Mat{};
}

class precalculated_matrix
    : public ::testing::Test
    , public ::testing::WithParamInterface<std::string> {
public:
    std::vector<cv::Mat> execute(const std::string& filename) {
        integral_computation executor;
        std::vector<cv::Mat> result;
        executor.set_on_complete([&](const auto& tasks) {
            for(const auto& task : tasks)
                result.push_back(task->get_result());
        });

        const auto input_path = fs::path(data_path) / (filename + ".txt");
        auto input = load_input_file(input_path.string());

        executor.enqueue_file(filename, input);
        executor.wait_for_complete();
        return result;
    }
};

static const std::string input_files[] = {
    "CV_16U_3x2x1", //
    "CV_8U_6x6x1", //
    "CV_8U_2x2x3", //
    "CV_8U_159x181x3",
};

TEST_P(precalculated_matrix, mat_is_equal) {
    const auto filename = GetParam();
    const auto expect_path = fs::path(data_path) / "expected";
    const auto expect_file =
        (expect_path / filename).replace_extension(".integral");
    const auto expect_mat = load_expect_file(expect_file.string());
    std::vector<cv::Mat> mats_by_channel(expect_mat.channels());
    cv::split(expect_mat, mats_by_channel);

    const auto computed_mats = execute(filename);
    ASSERT_NE(0, computed_mats.size());
    ASSERT_EQ(expect_mat.channels(), computed_mats.size());
    for(std::size_t i = 0; i < computed_mats.size(); ++i) {
        const auto& result = computed_mats[i];
        const auto& expect = mats_by_channel[i];

        cv::Mat cmp;
        cv::bitwise_xor(result, expect, cmp);
        ASSERT_EQ(cv::countNonZero(cmp), 0) << i;
    }
}

INSTANTIATE_TEST_CASE_P(
    , precalculated_matrix, ::testing::ValuesIn(input_files));

} // namespace
} // namespace test
} // namespace sp