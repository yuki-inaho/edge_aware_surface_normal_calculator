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

void showCVMat(std::string title, const cv::Mat &image)
{
    while (true)
    {
        cv::imshow(title, image);
        if (cv::waitKey(10) == 27 || cv::waitKey(10) == 'q')
            break;
    }
    cv::destroyAllWindows();
};

cv::Mat colorize_surface_normal(const cv::Mat& surface_normal){
    cv::Mat colorized_image = cv::Mat::zeros(cv::Size(surface_normal.cols, surface_normal.rows), CV_8UC3);

    for (size_t y = 0; y < surface_normal.rows; ++y)
    {
        for (size_t x = 0; x < surface_normal.cols; ++x)
        {
            if(surface_normal.at<cv::Vec3f>(y,x)[0] != 0 || surface_normal.at<cv::Vec3f>(y,x)[1] != 0 || surface_normal.at<cv::Vec3f>(y,x)[2] != -1){
                colorized_image.at<cv::Vec3b>(y,x)[0] = uchar(surface_normal.at<cv::Vec3f>(y,x)[0] * 255);
                colorized_image.at<cv::Vec3b>(y,x)[1] = uchar(surface_normal.at<cv::Vec3f>(y,x)[1] * 255);
                colorized_image.at<cv::Vec3b>(y,x)[2] = uchar(-1 * surface_normal.at<cv::Vec3f>(y,x)[2] * 255);
            }
        }
    }
    return colorized_image;
}

