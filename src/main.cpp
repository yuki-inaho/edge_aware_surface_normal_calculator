#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "argparse.hpp"
#include "toml_reader.h"
#include "depth_frame.h"
#include "surface_normal_calculator.h"
#include "util.h"

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

    // Set Camera Parameters
    CameraParameter camera_parameter = {
        cfg_params.ReadFloatData("Camera", "fx"),
        cfg_params.ReadFloatData("Camera", "fy"),
        cfg_params.ReadFloatData("Camera", "cx"),
        cfg_params.ReadFloatData("Camera", "cy"),
        cfg_params.ReadIntData("Camera", "width"),  //image_width
        cfg_params.ReadIntData("Camera", "height")  //image_height
    };

    // Set Depth Information
    DepthFrame depth_frame(depth_image, camera_parameter);
    SurfaceNormalCalculator surface_normal_calculator(camera_parameter);
    surface_normal_calculator.compute(depth_frame);
    cv::Mat edge_image = surface_normal_calculator.get_edge_image();
    cv::Mat surface_normal_image = surface_normal_calculator.get_surface_normal_image();    

    showCVMat("test", colorize_surface_normal(surface_normal_image));
    return 0;
}
