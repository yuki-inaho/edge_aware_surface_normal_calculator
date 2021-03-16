#pragma once

#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <numeric>

namespace fs = std::experimental::filesystem;

std::string concatDirectoryAndDataNames(std::string dir_name, std::string data_name)
{
    fs::path save_path = "";
    save_path.append(dir_name);
    save_path.append(data_name);
    return save_path.string<char>();
}

std::string getParentDir()
{
    // assume executive file is exist on {ParentDIR}/build
    fs::path parent_dir_path = fs::canonical("..");
    return parent_dir_path.string<char>();
}

bool mkdir(std::string dir_name)
{
    return fs::create_directory(dir_name);
}

void dumpCVMat(std::string name, const cv::Mat &image)
{
    cv::FileStorage fs(name, cv::FileStorage::WRITE);
    fs << "mat" << image;
};
