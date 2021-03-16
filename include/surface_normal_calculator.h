#pragma once
#include <opencv2/opencv.hpp>
#include "depth_frame.h"
#include "struct.h"

class SurfaceNormalCalculator
{
public:
    SurfaceNormalCalculator(CameraParameter camera_parameter, bool enable_edge_filter=true)
    {
        camera_parameter_ = camera_parameter;
        edge_image_ = cv::Mat::zeros(cv::Size(camera_parameter_.image_width, camera_parameter_.image_height), CV_8UC1);
        enable_edge_filter_ = enable_edge_filter;
    }

    void compute(const DepthFrame &frame){
        compute_spatial_difference(frame);
    };

    void compute_spatial_difference(const DepthFrame &frame, int32_t kernel_size = 3, float kernel_scale = 1./8., int32_t noise_reduction_kernel_size=3){
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
			const float sigma_space = 0.3*((noise_reduction_kernel_size-1)*0.5 - 1) + 0.8;
			cv::Mat depth_z_dx_prefilter = depth_image_z_dx_.clone();
			cv::bilateralFilter(depth_z_dx_prefilter, depth_image_z_dx_, noise_reduction_kernel_size, 10, sigma_space);
            cv::Mat depth_z_dy_prefilter = depth_image_z_dy_.clone();
			cv::bilateralFilter(depth_z_dy_prefilter, depth_image_z_dy_, noise_reduction_kernel_size, 10, sigma_space);
		}        
    }

private:
	cv::Mat depth_image_x_dx_, depth_image_y_dy_, depth_image_z_dx_, depth_image_z_dy_;  //Spatial differences of depth image
    cv::Mat edge_image_;
    
    CameraParameter camera_parameter_;
    bool enable_edge_filter_;
};