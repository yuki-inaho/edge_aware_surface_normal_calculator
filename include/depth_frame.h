
#pragma once
#include "struct.h"

class DepthFrame
{
public:
    DepthFrame(const cv::Mat &depth_image, const CameraParameter &camera_parameter)
    {
        camera_parameter_ = camera_parameter;
        SetDepthImage(depth_image);
    }

    cv::Mat get_depth_image() const {
        return depth_image_.clone();
    }

    // TODO: zero copy?
    cv::Mat get_x_image() const {
        return frustum_spatial_info_3d_.x_image.clone();
    }

    cv::Mat get_y_image() const {
        return frustum_spatial_info_3d_.y_image.clone();
    }

    cv::Mat get_z_image() const {
        return frustum_spatial_info_3d_.z_image.clone();
    }

    std::vector<float> get_center_point() const {
        std::vector<float> center_point = {center_x_, center_y_, center_z_};
        return center_point;
    }

private:
    void SetDepthImage(const cv::Mat &depth_image)
    {
        TypeValidationDepthImage(depth_image);
        depth_image_ = depth_image;
        frustum_spatial_info_3d_ = ConvertDepthImageToFrustumInformation(depth_image_);
    }

    void TypeValidationDepthImage(const cv::Mat depth_image)
    {
        if (depth_image.type() != CV_16UC1)
        {
            std::cerr << "depth_image.channels() != CV_16UC1" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if ((depth_image.rows != camera_parameter_.image_height) || (depth_image.cols != camera_parameter_.image_width))
        {
            std::cerr << "(depth_image.rows != camera_parameter_.image_height) || (depth_image.cols != camera_parameter_.image_width)" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    FrustumSpatialPointInformation ConvertDepthImageToFrustumInformation(const cv::Mat &depth_image)
    {
        cv::Mat x_image(camera_parameter_.image_height, camera_parameter_.image_width, CV_32FC1);
        cv::Mat y_image(camera_parameter_.image_height, camera_parameter_.image_width, CV_32FC1);
        cv::Mat z_image(camera_parameter_.image_height, camera_parameter_.image_width, CV_32FC1);

        float center_x = 0;
        float center_y = 0;
        float center_z = 0;
        int32_t count_non_zero = 0;
        for (size_t h = 0; h < depth_image.rows; h++)
        {
            float *x_ptr = (float *)x_image.ptr(h);
            float *y_ptr = (float *)y_image.ptr(h);
            float *z_ptr = (float *)z_image.ptr(h);
            for (size_t w = 0; w < depth_image.cols; w++)
            {
                float z_value = float(depth_image.at<uint16_t>(h, w)) / 1000.0f;
                if (z_value > 0 && std::isfinite(z_value))
                {
                    *z_ptr = z_value;
                    *x_ptr = z_value * (float(w) - camera_parameter_.cx) * (1.0 / camera_parameter_.fx);
                    *y_ptr = z_value * (float(h) - camera_parameter_.cy) * (1.0 / camera_parameter_.fy);
                    center_z += *z_ptr;
                    center_x += *x_ptr;
                    center_y += *y_ptr;
                    count_non_zero;
                }
                ++x_ptr;
                ++y_ptr;
                ++z_ptr;
            }
        }

        center_x /= count_non_zero;
        center_y /= count_non_zero;
        center_z /= count_non_zero;

        center_x_ = center_x;
        center_y_ = center_y;
        center_z_ = center_z;

        FrustumSpatialPointInformation frustum_spatial_info_3d = {x_image, y_image, z_image};
        return frustum_spatial_info_3d;
    }

    float center_x_;
    float center_y_;
    float center_z_;

    cv::Mat depth_image_;
    FrustumSpatialPointInformation frustum_spatial_info_3d_;
    CameraParameter camera_parameter_;
};