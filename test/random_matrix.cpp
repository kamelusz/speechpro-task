#include "common.hpp"

#include <unordered_map>
#include "integral_processing.hpp"
#include "opencv2/imgproc.hpp"

namespace sp {
namespace test {
namespace {

struct param_t {
    int rows;
    int cols;
    int channels;
    int depth;
    double low;
    double high;
};

class random_matrix
    : public ::testing::Test
    , public ::testing::WithParamInterface<param_t> {
public:
    void SetUp() override;

protected:
    cv::Mat random_mat;
    cv::Mat integral_mat;
};

void random_matrix::SetUp() {
    const auto channels = GetParam().channels;
    std::unordered_map<int, int> type_map{
        std::make_pair(CV_8U, CV_8UC(channels)),
        std::make_pair(CV_16U, CV_16UC(channels)),
        std::make_pair(CV_16S, CV_16SC(channels)),
    };

    const auto depth = GetParam().depth;
    ASSERT_TRUE(!!type_map.count(depth));
    const auto type = type_map[depth];
    const auto rows = GetParam().rows;
    const auto cols = GetParam().cols;
    random_mat.create(rows, cols, type);

    const auto l = GetParam().low;
    const auto h = GetParam().high;
    std::unordered_map<int, std::pair<cv::Scalar, cv::Scalar>> scalars = {
        std::make_pair(1, std::make_pair(cv::Scalar(l), cv::Scalar(h))),
        std::make_pair(2, std::make_pair(cv::Scalar(l, l), cv::Scalar(h, h))),
        std::make_pair(3, std::make_pair(cv::Scalar(l, l, l), cv::Scalar(h, h, h))),
    };
    ASSERT_TRUE(!!scalars.count(channels));
    const auto& scalar = scalars[channels];
    cv::randu(random_mat, scalar.first, scalar.second);

    cv::integral(random_mat, integral_mat, CV_64F);
    integral_mat(cv::Range(1, rows + 1), cv::Range(1, cols + 1))
        .copyTo(integral_mat);
}

static const param_t params[] = {
    {10, 10, 1, CV_8U, 0, 100}, //
    {10, 10, 2, CV_8U, 0, 100}, //
    {10, 10, 3, CV_8U, 0, 100}, //
    {100, 100, 1, CV_8U, 0, 100}, //
    {100, 100, 2, CV_8U, 0, 100}, //
    {100, 100, 3, CV_8U, 0, 100}, //
    {1'001, 10'001, 1, CV_16U, 0, 3'400}, //
    {1'001, 10'001, 2, CV_16U, 0, 100}, //
    {1'001, 1'0001, 3, CV_16U, 0, 100}, //
    {1'001, 10'001, 1, CV_16S, -100'000, 100'000}, //
    {1'001, 10'001, 2, CV_16S, -100'000, 100'000}, //
    {1'001, 10'001, 3, CV_16S, -100'000, 100'000}, //
    {3'141, 278, 1, CV_16S, -123'456, 123'456}, //
    {3'141, 278, 2, CV_16S, -123'456, 123'456}, //
    {3'141, 278, 3, CV_16S, -123'456, 123'456}, //
    {4'096, 2'560, 1, CV_16S, -3'000'000, 3'000'000}, //
    {4'096, 2'560, 2, CV_16S, -3'000'000, 3'000'000}, //
    {4'096, 2'560, 3, CV_16S, -3'000'000, 3'000'000}, //
};

TEST_P(random_matrix, mat_is_equal) {
    const auto channels = integral_mat.channels();
    std::vector<cv::Mat> mats_by_channel(channels);
    cv::split(integral_mat, mats_by_channel);

    integral_computation executor;
    std::vector<cv::Mat> computed_mats;
    executor.set_on_complete([&](const auto& tasks) {
        for(const auto& task : tasks)
            computed_mats.push_back(task->get_result());
    });
    executor.enqueue_file("random", random_mat);
    executor.wait_for_complete();

    ASSERT_EQ(integral_mat.channels(), computed_mats.size());
    for(std::size_t i = 0; i < computed_mats.size(); ++i) {
        const auto& result = computed_mats[i];
        const auto& expect = mats_by_channel[i];

        cv::Mat cmp;
        cv::bitwise_xor(result, expect, cmp);
        ASSERT_EQ(cv::countNonZero(cmp), 0);
    }
}

INSTANTIATE_TEST_CASE_P(, random_matrix, ::testing::ValuesIn(params));

} // namespace
} // namespace test
} // namespace sp