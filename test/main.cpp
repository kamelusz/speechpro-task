#include "common.hpp"

#include "boost/program_options.hpp"

namespace sp {
namespace test {

std::string data_path;

} // namespace test
} // namespace sp

namespace opt = boost::program_options;

int main(int argc, char* argv[]) {
    opt::options_description desc;
    desc.add_options()(
        "data", opt::value<std::string>()->default_value("./data"),
        "test data directory");
    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    sp::test::data_path = vm["data"].as<std::string>();

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}