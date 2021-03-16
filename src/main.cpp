#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "argparse.hpp"
#include "toml_reader.h"
#include "depth_frame.h"
#include "util.h"

using namespace std;
using namespace cv;

std::string WORK_DIR = getParentDir();

struct ParsedArgument
{
    std::string config_file_path;
};

ParsedArgument parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser("edge_aware_surface_normal_calculator", "Calculate surface normal");
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

    // Get target depth image
    std::string data_dir_path = concatDirectoryAndDataNames(WORK_DIR, cfg_params.ReadStringData("Main", "data_dir"));
    std::string image_path = concatDirectoryAndDataNames(data_dir_path, cfg_params.ReadStringData("Main", "image_name"));

    std::cout << "Input Image: " << image_path << std::endl;
    cv::Mat depth_image = cv::imread(image_path, cv::IMREAD_ANYDEPTH);
    std::cout << "width:" << depth_image.cols << " "
              << "height:" << depth_image.rows << endl;

    CameraParameter camera_parameter = {
        cfg_params.ReadFloatData("Camera", "fx"),
        cfg_params.ReadFloatData("Camera", "fy"),
        cfg_params.ReadFloatData("Camera", "cx"),
        cfg_params.ReadFloatData("Camera", "cy"),
        cfg_params.ReadIntData("Camera", "width"),
        cfg_params.ReadIntData("Camera", "height")
    };

    DepthFrame depth_frame(depth_image, camera_parameter);

    //showCVMat("test", depth_image);

    return 0;
}
