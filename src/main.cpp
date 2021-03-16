#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "argparse.hpp"
#include "toml_reader.h"
#include "util.h"

using namespace std;
using namespace cv;

std::string PARENT_DIR = getParentDir();

struct ParsedArgument
{
    std::string config_file_path;
};

ParsedArgument parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser("edge_aware_surface_normal_calculator", "Calculate surface normal", "MIT License");
    parser.addArgument({"--config-file-path", "-c"}, "Relative location of config file");
    auto args = parser.parseArgs(argc, argv);
    std::string config_file_path = args.safeGet<std::string>("config-file-path", "../config/config.toml");
    ParsedArgument parsed_args = {config_file_path};
    return parsed_args;
}

int main(int argc, char **argv)
{
    ParsedArgument args = parse_args(argc, argv);
    TomlReader cfg_params(args.config_file_path);
    std::string image_path = concatDirectoryAndDataNames(cfg_params.ReadStringData("Main", "data_dir"), cfg_params.ReadStringData("Main", "image_name"));
    cv::Mat depth_image = cv::imread(image_path, cv::IMREAD_ANYDEPTH);

    return 0;
}
