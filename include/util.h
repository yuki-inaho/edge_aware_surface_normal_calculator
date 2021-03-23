#pragma once

#include <opencv2/opencv.hpp>

namespace fs = std::experimental::filesystem;

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
