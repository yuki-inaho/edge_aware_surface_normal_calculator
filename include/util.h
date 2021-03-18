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

cv::Mat ColorizeSurfaceNormal(const cv::Mat &surface_normal)
{
    cv::Mat colorized_image = cv::Mat::zeros(cv::Size(surface_normal.cols, surface_normal.rows), CV_8UC3);

    for (size_t y = 0; y < surface_normal.rows; ++y)
    {
        for (size_t x = 0; x < surface_normal.cols; ++x)
        {
            if (surface_normal.at<cv::Vec3f>(y, x)[0] != 0 || surface_normal.at<cv::Vec3f>(y, x)[1] != 0 || surface_normal.at<cv::Vec3f>(y, x)[2] != -1)
            {
                if (surface_normal.at<cv::Vec3f>(y, x)[2] == 0)
                    continue;
                colorized_image.at<cv::Vec3b>(y, x)[0] = uchar(surface_normal.at<cv::Vec3f>(y, x)[0] * 255);
                colorized_image.at<cv::Vec3b>(y, x)[1] = uchar(surface_normal.at<cv::Vec3f>(y, x)[1] * 255);
                colorized_image.at<cv::Vec3b>(y, x)[2] = uchar(-1 * surface_normal.at<cv::Vec3f>(y, x)[2] * 255);
            }
        }
    }
    return colorized_image;
}

void FillZeroFarDepth(cv::Mat &depth_image, uint16_t slope = 2000)
{
    for (int v = 0; v < depth_image.rows; v++)
    {
        for (int u = 0; u < depth_image.cols; u++)
        {
            //if(depth_image.at<unsigned short>(v, u) == 65535)
            if (depth_image.at<unsigned short>(v, u) > slope)
                depth_image.at<unsigned short>(v, u) = 0;
        }
    }
}

cv::Mat ColorizeDepthImage(const cv::Mat &depth_image, uint16_t slope = 2000)
{
    cv::Mat depth_image_8uc1, colorized_depth_image;
    depth_image.convertTo(depth_image_8uc1, CV_8U, 255.0 / slope);
    cv::applyColorMap(depth_image_8uc1, colorized_depth_image, cv::COLORMAP_JET);
    return colorized_depth_image;
}

void DrawMultipleImages(std::vector<std::string> names, std::vector<cv::Mat> images, int32_t text_area_bold = 30, float resized_rate = 0.9)
{
    int32_t n_images = images.size();
    int32_t width_whole = 0;
    int32_t height_max = 0;
    for (auto &image : images)
    {
        width_whole += image.cols;
        height_max = std::max(height_max, image.rows);
    }
    int32_t width_point = 0;
    cv::Mat image_concat = cv::Mat::zeros(cv::Size(width_whole, height_max + text_area_bold), CV_8UC3);
    int32_t image_count = 0;
    for (auto &image : images)
    {
        cv::Mat aux = image_concat.colRange(width_point, width_point + image.cols).rowRange(text_area_bold, text_area_bold + image.rows);
        cv::putText(image_concat, names[image_count], cv::Point(width_point, text_area_bold - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, CV_AA);
        image.copyTo(aux);
        width_point += image.cols;
        image_count++;
    }
    cv::Mat image_concat_resized;
    cv::resize(image_concat, image_concat_resized, cv::Size(), resized_rate, resized_rate);
    showCVMat("concat", image_concat_resized);
}
