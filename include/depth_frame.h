
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
                }
                ++x_ptr;
                ++y_ptr;
                ++z_ptr;
            }
        }

        FrustumSpatialPointInformation frustum_spatial_info_3d = {x_image, y_image, z_image};
        return frustum_spatial_info_3d;
    }

    cv::Mat depth_image_;
    FrustumSpatialPointInformation frustum_spatial_info_3d_;
    CameraParameter camera_parameter_;
};