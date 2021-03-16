#pragma once
#include <opencv2/opencv.hpp>
#include "depth_frame.h"
#include "struct.h"

class SurfaceNormalCalculator
{
public:
    SurfaceNormalCalculator(CameraParameter camera_parameter, bool enable_edge_filter = true)
    {
        camera_parameter_ = camera_parameter;
        edge_image_ = cv::Mat::zeros(cv::Size(camera_parameter_.image_width, camera_parameter_.image_height), CV_8UC1);
        enable_edge_filter_ = enable_edge_filter;
    }

    void compute(const DepthFrame &frame)
    {
        ComputeSpatialDifference(frame);
        ComputeDepthEdge(frame);
        NonMaximumSuppression();
    };

    cv::Mat get_edge_image(){
        return edge_;
    }

private:
    cv::Mat depth_image_x_dx_, depth_image_y_dy_, depth_image_z_dx_, depth_image_z_dy_; //Spatial differences of depth image
    cv::Mat edge_image_;

    void ComputeSpatialDifference(const DepthFrame &frame, int32_t kernel_size = 3, float kernel_scale = 1. / 8., int32_t noise_reduction_kernel_size = 3)
    {
        // TODO: no copy
        cv::Mat x_image = frame.get_x_image();
        cv::Mat y_image = frame.get_y_image();
        cv::Mat z_image = frame.get_z_image();

        cv::Sobel(x_image, depth_image_x_dx_, -1, 1, 0, kernel_size, kernel_scale);
        cv::Sobel(y_image, depth_image_y_dy_, -1, 0, 1, kernel_size, kernel_scale);
        cv::Sobel(z_image, depth_image_z_dx_, -1, 1, 0, kernel_size, kernel_scale);
        cv::Sobel(z_image, depth_image_z_dy_, -1, 0, 1, kernel_size, kernel_scale);
        if (enable_edge_filter_)
        {
            const float sigma_space = 0.3 * ((noise_reduction_kernel_size - 1) * 0.5 - 1) + 0.8;
            cv::Mat depth_z_dx_prefilter = depth_image_z_dx_.clone();
            cv::bilateralFilter(depth_z_dx_prefilter, depth_image_z_dx_, noise_reduction_kernel_size, 10, sigma_space);
            cv::Mat depth_z_dy_prefilter = depth_image_z_dy_.clone();
            cv::bilateralFilter(depth_z_dy_prefilter, depth_image_z_dy_, noise_reduction_kernel_size, 10, sigma_space);
        }
    }

    void ComputeDepthEdge(const DepthFrame &frame, float depth_threshold_factor = 0.01)
    {
        cv::Mat z_image = frame.get_z_image();
        cv::Mat edge = cv::Mat::zeros(camera_parameter_.image_height, camera_parameter_.image_width, CV_8UC1);
        for (int v = 0; v < depth_image_z_dx_.rows; ++v)
        {
            for (int u = 0; u < depth_image_z_dx_.cols; ++u)
            {
                float depth = z_image.at<float>(v, u);
                edge.at<uchar>(v, u) = 0;
                if (depth == 0.f)
                    continue;
                float edge_threshold = std::max(0.0f, depth_threshold_factor * depth); // is it correct?
                float z_dx = depth_image_z_dx_.at<float>(v, u);
                float z_dy = depth_image_z_dy_.at<float>(v, u);
                if (z_dx <= -edge_threshold || z_dx >= edge_threshold || z_dy <= -edge_threshold || z_dy >= edge_threshold)
                    edge.at<uchar>(v, u) = (uchar)std::min<float>(255.f, 50.f * (1. + sqrt(z_dx * z_dx + z_dy * z_dy)));
            }
        }
        edge_ = edge;
    }

    void NonMaximumSuppression()
    {
        cv::Mat edge_refined = cv::Mat::zeros(edge_.rows, edge_.cols, CV_8UC1);
        for (int v = 1; v < edge_refined.rows - 1; ++v)
        {
            for (int u = 1; u < edge_refined.cols - 1; ++u)
            {
                if (edge_.at<uchar>(v, u) == 0)
                    continue;

                float dx = depth_image_z_dx_.at<float>(v, u);
                float dy = depth_image_z_dy_.at<float>(v, u);
                if (dx == 0 && dy == 0)
                    continue;
                double slope_norm = sqrt(dx * dx + dy * dy);
                dx = floor(dx / slope_norm + 0.5);
                dy = floor(dy / slope_norm + 0.5);
                uchar &edge_uv = edge_.at<uchar>(v, u);
                if (edge_uv >= edge_.at<uchar>(v + dy, u + dx) && edge_uv >= edge_.at<uchar>(v - dy, u - dx))
                    edge_uv = (uchar)254;  // 254?
                else
                    edge_refined.at<uchar>(v, u) = 1; // 1?
            }
        }


        for (int v = 1; v < edge_.rows - 1; ++v)
            for (int u = 1; u < edge_.cols - 1; ++u)
                if (edge_refined.at<uchar>(v, u) == 1)
                    edge_.at<uchar>(v, u) = 0;
    }

    CameraParameter camera_parameter_;
    bool enable_edge_filter_;
    cv::Mat edge_;
};