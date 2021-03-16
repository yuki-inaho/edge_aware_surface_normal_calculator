#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

struct CameraParameter{
    float fx;
    float fy;
    float cx;
    float cy;
    int32_t image_width;
    int32_t image_height;
};

struct FrustumSpatialPointInformation{
    cv::Mat x_image;  // horisontal
    cv::Mat y_image;  // vertical
    cv::Mat z_image;  // depth-wise
};