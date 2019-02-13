///
/// \file
/// Defines the context of separated channel integral image computation.
///

#pragma once
#include <memory>
#include <string>
#include "opencv2/imgproc.hpp"

namespace sp {

/** @brief Incapsulates separated channel data for integral computation.

It is worth mentioning that, manually splitting matrix to several channels
isn't requred. Instead that channel number for computation can be specified.
Class ensures thread safety for the case of computation on the same input
matrix in different threads.

Usage example:
@code
    auto first_channel = processing_context("Lena", LenaMat, 1);
    first_channel.execute();
    const auto& result = first_channel.get_result();
@endcode
*/
class processing_context {
public:
    using ptr = std::unique_ptr<processing_context>;

public:
    /** @brief Constructs context for integral image computation.

    @note In case of unspecified channel, just first channel (at index 0) will
    be computed.
    @param id Identification for matrix.
    @param mat Input matrix.
    @param channel Channel of image matrix to compute.
    */
    processing_context(
        const std::string& id, const cv::Mat& mat, int channel = 0)
        : id(id), image(mat), channel(channel) {
        result.create(mat.size(), CV_64F);
    }

public:
    //! @brief Retruns identification of matrix.
    const std::string& get_id() const noexcept;

    //! @brief Retruns processed channel number.
    int get_channel() const noexcept;

    //! @brief Retruns computed matrix.
    const cv::Mat& get_result() const noexcept;

    //! @brief Retruns original matrix.
    const cv::Mat& get_image() const noexcept;

    //! @brief Computes integral image
    void execute();

private:
    std::string id;
    cv::Mat image;
    int channel;
    cv::Mat result;
};

} // namespace sp