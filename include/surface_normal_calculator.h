/*!
 Some part of below code is borrowed from 
 https://github.com/ipa320/cob_object_perception/blob/indigo_dev/cob_surface_classification/common/include/cob_surface_classification/edge_detection.h

 *****************************************************************
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer. \n
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. \n
 * - Neither the name of the Fraunhofer Institute for Manufacturing
 * Engineering and Automation (IPA) nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission. \n
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License LGPL as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License LGPL for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License LGPL along with this program.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

#pragma once
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <eigen3/Eigen/Dense>
#include "thinning.hpp"
#include "depth_frame.h"
#include "struct.h"
#include "fast_arithmetic.h"

#define MIN_DISTANCE_TO_DEPTH_EDGE 1
#define MIN_SCAN_LINE_WIDTH_FRACTION_FROM_MAX 1

class SurfaceNormalCalculator
{
public:
    SurfaceNormalCalculator(CameraParameter camera_parameter, bool enable_adaptive_scan_line = true, bool enable_edge_filter = true, int32_t scan_line_width_at_2m = 15)
    {
        omp_init_lock(&omp_lock_);
        camera_parameter_ = camera_parameter;
        edge_image_ = cv::Mat::zeros(cv::Size(camera_parameter_.image_width, camera_parameter_.image_height), CV_8UC1);
        enable_edge_filter_ = enable_edge_filter;
        enable_adaptive_scan_line_ = enable_adaptive_scan_line;

        // TODO: Add explanation
        scan_line_model_n_ = (20. - scan_line_width_at_2m) / (double)3.0;
        scan_line_model_m_ = 10 - 2 * scan_line_model_n_;
    }

    ~SurfaceNormalCalculator()
    {
        omp_destroy_lock(&omp_lock_);
    }

    void compute(const DepthFrame &frame)
    {
        ComputeSpatialDifference(frame);
        ComputeDepthEdge(frame);
        NonMaximumSuppression();
        thinning(edge_);

        if (!enable_adaptive_scan_line_)
            cv::integral(edge_, edge_integral_, CV_32S);

        ComputeIntegralImageHorisontal(depth_image_x_dx_, depth_image_x_dx_integral_horisontal_, depth_image_z_dx_, depth_image_z_dx_integral_horisontal_);
        if (enable_adaptive_scan_line_)
            ComputeEdgeDistanceMapHorizontal(edge_, distance_map_horizontal_);
        ScanLineHorisontal(frame, distance_map_horizontal_);

        // TODO: confirm
        cv::Mat depth_image_y_dy_transposed = depth_image_y_dy_.clone().t();
        cv::Mat depth_image_z_dy_transposed = depth_image_z_dy_.clone().t();

        ComputeIntegralImageHorisontal(depth_image_y_dy_transposed, depth_image_y_dy_integral_vertical_, depth_image_z_dy_transposed, depth_image_z_dy_integral_vertical_);
        if (enable_adaptive_scan_line_)
            ComputeEdgeDistanceMapVertical(edge_, distance_map_vertical_);
        ScanLineVertical(frame, distance_map_vertical_);
        UpdateEdge();
        if (enable_adaptive_scan_line_)
        {
            ComputeEdgeDistanceMapHorizontal(edge_, distance_map_horizontal_);
            ComputeEdgeDistanceMapVertical(edge_, distance_map_vertical_);
        }
        else
        {
            cv::integral(edge_, edge_integral_, CV_32S);
        }
        ComputeSurfaceNormal(frame);
    };

    cv::Mat get_edge_image()
    {
        return edge_.clone();
    }

    cv::Mat get_surface_normal_image()
    {
        return normal_.clone();
    }

private:
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
                    edge_uv = (uchar)254;
                else
                    edge_refined.at<uchar>(v, u) = 1;
            }
        }

        for (int v = 1; v < edge_.rows - 1; ++v)
            for (int u = 1; u < edge_.cols - 1; ++u)
                if (edge_refined.at<uchar>(v, u) == 1)
                    edge_.at<uchar>(v, u) = 0;
    }

    void ComputeIntegralImageHorisontal(const cv::Mat &srcX, cv::Mat &dstX, const cv::Mat &srcZ, cv::Mat &dstZ)
    {
        dstX = cv::Mat(srcX.rows, srcX.cols, CV_32FC1);
        dstZ = cv::Mat(srcX.rows, srcX.cols, CV_32FC1);
        for (int v = 0; v < srcX.rows; ++v)
        {
            const float *srcX_ptr = (const float *)srcX.ptr(v);
            const float *srcZ_ptr = (const float *)srcZ.ptr(v);
            float *dstX_ptr = (float *)dstX.ptr(v);
            float *dstZ_ptr = (float *)dstZ.ptr(v);
            float sumX = 0.f;
            float sumZ = 0.f;
            for (int u = 0; u < srcX.cols; ++u)
            {
                sumX += *srcX_ptr;
                sumZ += *srcZ_ptr;
                *dstX_ptr = sumX;
                srcX_ptr++;
                dstX_ptr++;
                *dstZ_ptr = sumZ;
                srcZ_ptr++;
                dstZ_ptr++;
            }
        }
    }

    void ComputeIntegralImageVertical(const cv::Mat &srcY, cv::Mat &dstY, const cv::Mat &srcZ, cv::Mat &dstZ)
    {
        dstY = cv::Mat(srcY.rows, srcY.cols, CV_32FC1);
        dstZ = cv::Mat(srcY.rows, srcY.cols, CV_32FC1);
        float *dstY_ptr = (float *)dstY.ptr(0);
        const float *srcY_ptr = (const float *)srcY.ptr(0);
        float *dstZ_ptr = (float *)dstZ.ptr(0);
        const float *srcZ_ptr = (const float *)srcZ.ptr(0);
        for (int u = 0; u < srcY.cols; ++u)
        {
            *dstY_ptr = *srcY_ptr;
            dstY_ptr++;
            srcY_ptr++;
            *dstZ_ptr = *srcZ_ptr;
            dstZ_ptr++;
            srcZ_ptr++;
        }
        for (int v = 1; v < srcY.rows; ++v)
        {
            const float *srcY_ptr = (const float *)srcY.ptr(v);
            const float *srcZ_ptr = (const float *)srcZ.ptr(v);
            float *dstY_ptr = (float *)dstY.ptr(v);
            float *dstYprev_ptr = (float *)dstY.ptr(v - 1);
            float *dstZ_ptr = (float *)dstZ.ptr(v);
            float *dstZprev_ptr = (float *)dstZ.ptr(v - 1);
            for (int u = 0; u < srcY.cols; ++u)
            {
                if (*srcY_ptr > 0.f)
                {
                    *dstY_ptr = *dstYprev_ptr + *srcY_ptr;
                    *dstZ_ptr = *dstZprev_ptr + *srcZ_ptr;
                }
                else
                {
                    *dstY_ptr = *dstYprev_ptr;
                    *dstZ_ptr = *dstZprev_ptr;
                }
                srcY_ptr++;
                dstY_ptr++;
                dstYprev_ptr++;
                srcZ_ptr++;
                dstZ_ptr++;
                dstZprev_ptr++;
            }
        }
    }

    void ComputeEdgeDistanceMapHorizontal(const cv::Mat &edge, cv::Mat &distance_map)
    {
        distance_map.create(edge.rows, edge.cols, CV_32SC2);
        for (int v = 0; v < edge.rows; ++v)
        {
            distance_map.at<cv::Vec2i>(v, 0).val[0] = 0;
            for (int u = 1; u < edge.cols; ++u)
            {
                if (edge.at<uchar>(v, u) != 0)
                    distance_map.at<cv::Vec2i>(v, u).val[0] = 0;
                else
                    distance_map.at<cv::Vec2i>(v, u).val[0] = distance_map.at<cv::Vec2i>(v, u - 1).val[0] + 1;
            }
            distance_map.at<cv::Vec2i>(v, edge.cols - 1).val[1] = 0;
            for (int u = edge.cols - 2; u >= 0; --u)
            {
                if (edge.at<uchar>(v, u) != 0)
                    distance_map.at<cv::Vec2i>(v, u).val[1] = 0;
                else
                    distance_map.at<cv::Vec2i>(v, u).val[1] = distance_map.at<cv::Vec2i>(v, u + 1).val[1] + 1;
            }
        }
    }

    void ComputeEdgeDistanceMapVertical(const cv::Mat &edge, cv::Mat &distance_map)
    {
        distance_map.create(edge.rows, edge.cols, CV_32SC2);
        const int width = edge.cols;
        int *data = distance_map.ptr<int>(0);
        int *data_prev = distance_map.ptr<int>(0);
        const uchar *edge_ptr = edge.ptr<uchar>(1);
        for (int u = 0; u < width; ++u, data += 2)
            *data = 0;
        for (int v = 1; v < edge.rows; ++v)
        {
            for (int u = 0; u < width; ++u)
            {
                if (*edge_ptr != 0)
                    *data = 0;
                else
                    *data = *data_prev + 1;
                data += 2;
                data_prev += 2;
                ++edge_ptr;
            }
        }
        data = distance_map.ptr<int>(edge.rows - 1) + 2 * width - 1;
        data_prev = distance_map.ptr<int>(edge.rows - 1) + 2 * width - 1;
        edge_ptr = edge.ptr<uchar>(edge.rows - 1) - 1;
        for (int u = edge.cols - 1; u >= 0; --u, data -= 2)
            *data = 0;
        for (int v = edge.rows - 2; v >= 0; --v)
        {
            for (int u = edge.cols - 1; u >= 0; --u)
            {
                if (*edge_ptr != 0)
                    *data = 0;
                else
                    *data = *data_prev + 1;
                data -= 2;
                data_prev -= 2;
                --edge_ptr;
            }
        }
    }

    // scan_line_length_1 = left or above,  scan_line_length_2 = right or below
    inline bool AdaptScanLine(int &scan_line_length_1, int &scan_line_length_2, const cv::Mat &distance_map, const int u, const int v, const int min_line_width)
    {
        const int max_1 = scan_line_length_1;
        const int max_2 = scan_line_length_2;
        const cv::Vec2i &free_dist = distance_map.at<cv::Vec2i>(v, u);
        scan_line_length_1 = std::min(scan_line_length_1, free_dist.val[0] - 1 - MIN_DISTANCE_TO_DEPTH_EDGE);
        scan_line_length_2 = std::min(scan_line_length_2, free_dist.val[1] - 1 - MIN_DISTANCE_TO_DEPTH_EDGE);

        if ((scan_line_length_1 < min_line_width || MIN_SCAN_LINE_WIDTH_FRACTION_FROM_MAX * scan_line_length_1 < max_1) ||
            (scan_line_length_2 < min_line_width || MIN_SCAN_LINE_WIDTH_FRACTION_FROM_MAX * scan_line_length_2 < max_2))
            return false;
        return true;
    }

    // scan_line_length_1 = left or above,  scan_line_length_2 = right or below
    inline bool AdaptScanLineNormal(int &scan_line_length_1, int &scan_line_length_2, const cv::Mat &distance_map, const int u, const int v, const int min_line_width)
    {
        const int max_1 = scan_line_length_1;
        const int max_2 = scan_line_length_2;
        const cv::Vec2i &free_dist = distance_map.at<cv::Vec2i>(v, u);
        scan_line_length_1 = std::min(scan_line_length_1, free_dist.val[0] - 1 - MIN_DISTANCE_TO_DEPTH_EDGE);
        scan_line_length_2 = std::min(scan_line_length_2, free_dist.val[1] - 1 - MIN_DISTANCE_TO_DEPTH_EDGE);
        if ((scan_line_length_1 + scan_line_length_2 + 1) < min_line_width)
            return false;
        return true;
    }

    bool AdaptScanLineWidth(int &scan_line_width_left, int &scan_line_width_right, const cv::Mat &edge, const int u, const int v, const int min_line_width)
    {
        const int min_distance_to_depth_edge = MIN_DISTANCE_TO_DEPTH_EDGE;
        const int max_l = scan_line_width_left;
        for (int du = 0; du <= max_l + 1 + min_distance_to_depth_edge; ++du)
        {
            if (edge.at<uchar>(v, u - du) != 0)
            {
                scan_line_width_left = du - 1 - min_distance_to_depth_edge;
                if (scan_line_width_left < min_line_width || MIN_SCAN_LINE_WIDTH_FRACTION_FROM_MAX * scan_line_width_left < max_l)
                    return false;
                break;
            }
        }
        const int max_r = scan_line_width_right;
        for (int du = 0; du <= max_r + 1 + min_distance_to_depth_edge; ++du)
        {
            if (edge.at<uchar>(v, u + du) != 0)
            {
                scan_line_width_right = du - 1 - min_distance_to_depth_edge;
                if (scan_line_width_right < min_line_width || MIN_SCAN_LINE_WIDTH_FRACTION_FROM_MAX * scan_line_width_right < max_r)
                    return false;
                break;
            }
        }
        return true;
    }

    bool AdaptScanLineHeight(int &scan_line_height_upper, int &scan_line_height_lower, const cv::Mat &edge, const int u, const int v, const int min_line_width)
    {
        const int min_distance_to_depth_edge = MIN_DISTANCE_TO_DEPTH_EDGE;
        const int max_u = scan_line_height_upper;
        for (int dv = 0; dv <= max_u + 1 + min_distance_to_depth_edge; ++dv)
        {
            if (edge.at<uchar>(v - dv, u) != 0)
            {
                scan_line_height_upper = dv - 1 - min_distance_to_depth_edge;
                if (scan_line_height_upper < min_line_width || MIN_SCAN_LINE_WIDTH_FRACTION_FROM_MAX * scan_line_height_upper < max_u)
                    return false;
                break;
            }
        }
        const int max_l = scan_line_height_lower;
        for (int dv = 0; dv <= max_l + 1 + min_distance_to_depth_edge; ++dv)
        {
            if (edge.at<uchar>(v + dv, u) != 0)
            {
                scan_line_height_lower = dv - 1 - min_distance_to_depth_edge;
                if (scan_line_height_lower < min_line_width || MIN_SCAN_LINE_WIDTH_FRACTION_FROM_MAX * scan_line_height_lower < max_l)
                    return false;
                break;
            }
        }
        return true;
    }

    bool AdaptScanLineWidthNormal(int &scan_line_width_left, int &scan_line_width_right, const cv::Mat &edge, const int u, const int v, const int min_line_width)
    {
        const int min_distance_to_depth_edge = 2;
        const int max_l = scan_line_width_left;
        for (int du = 0; du <= max_l + 1 + min_distance_to_depth_edge; ++du)
        {
            if (edge.at<uchar>(v, u - du) != 0)
            {
                scan_line_width_left = du - 1 - min_distance_to_depth_edge;
                break;
            }
        }
        const int max_r = scan_line_width_right;
        for (int du = 0; du <= max_r + 1 + min_distance_to_depth_edge; ++du)
        {
            if (edge.at<uchar>(v, u + du) != 0)
            {
                scan_line_width_right = du - 1 - min_distance_to_depth_edge;
                break;
            }
        }
        if ((scan_line_width_right + scan_line_width_left) < min_line_width)
            return false;
        return true;
    }

    bool AdaptScanLineHeightNormal(int &scan_line_height_upper, int &scan_line_height_lower, const cv::Mat &edge, const int u, const int v, const int min_line_width)
    {
        const int min_distance_to_depth_edge = 2;
        const int max_u = scan_line_height_upper;
        for (int dv = 0; dv <= max_u + 1 + min_distance_to_depth_edge; ++dv)
        {
            if (edge.at<uchar>(v - dv, u) != 0)
            {
                scan_line_height_upper = dv - 1 - min_distance_to_depth_edge;
                break;
            }
        }
        const int max_l = scan_line_height_lower;
        for (int dv = 0; dv <= max_l + 1 + min_distance_to_depth_edge; ++dv)
        {
            if (edge.at<uchar>(v + dv, u) != 0)
            {
                scan_line_height_lower = dv - 1 - min_distance_to_depth_edge;
                break;
            }
        }
        if ((scan_line_height_lower + scan_line_height_upper) < min_line_width)
            return false;
        return true;
    }

    void ScanLineHorisontal(const DepthFrame &frame, cv::Mat &distance_map_horizontal, int32_t max_scan_line_width = 20, int32_t min_scan_line_width = 1, int32_t scan_line_width_init = 10, int32_t min_detectable_edge_angle = 45)
    {
        // TODO: zero copy?
        cv::Mat z_image = frame.get_z_image();
        const int max_v = depth_image_z_dx_.rows - max_scan_line_width - 2;
        const int max_u = depth_image_z_dx_.cols - max_scan_line_width - 2;

#pragma omp parallel for
        for (int v = max_scan_line_width + 1; v < max_v; ++v)
        {
            int scan_line_width = scan_line_width_init;
            int last_line_width = scan_line_width;
            int edge_start_index = -1;
            float max_edge_strength = 0;
            for (int u = max_scan_line_width + 1; u < max_u; ++u)
            {
                const float depth = z_image.at<float>(v, u);
                if (depth == 0.f)
                    continue;

                // depth dependent scan line width for slope computation (1px width per 0.10m depth)
                scan_line_width = std::min(int(scan_line_model_m_ * depth + scan_line_model_n_), max_scan_line_width);
                if (scan_line_width <= min_scan_line_width)
                    scan_line_width = last_line_width;
                else
                    last_line_width = scan_line_width;

                int scan_line_width_left = scan_line_width;
                int scan_line_width_right = scan_line_width;

                if (enable_adaptive_scan_line_)
                {
                    if (AdaptScanLine(scan_line_width_left, scan_line_width_right, distance_map_horizontal, u, v, min_scan_line_width) == false)
                    {
                        edge_start_index = -1;
                        max_edge_strength = 0.f;
                        continue;
                    }
                }
                else
                {
                    if (edge_integral_.at<int>(v + 1, u + scan_line_width + 1) - edge_integral_.at<int>(v + 1, u - scan_line_width - 2) - edge_integral_.at<int>(v, u + scan_line_width + 1) + edge_integral_.at<int>(v, u - scan_line_width - 2) != 0)
                    {
                        edge_start_index = -1;
                        max_edge_strength = 0.f;
                        continue;
                    }
                }

                // get average differences in x and z direction (ATTENTION: the integral images provide just the sum, not divided by number of elements, however, further processing only needs the sum, not the real average)
                // remark: the indexing of the integral image here differs from the OpenCV definition (here: the value of a cell is included in the sum of the integral image's cell)
                const float avg_dx_l = depth_image_x_dx_integral_horisontal_.at<float>(v, u - 1) - depth_image_x_dx_integral_horisontal_.at<float>(v, u - scan_line_width_left);
                const float avg_dx_r = depth_image_x_dx_integral_horisontal_.at<float>(v, u + scan_line_width_right) - depth_image_x_dx_integral_horisontal_.at<float>(v, u + 1);
                const float avg_dz_l = depth_image_z_dx_integral_horisontal_.at<float>(v, u - 1) - depth_image_z_dx_integral_horisontal_.at<float>(v, u - scan_line_width_left);
                const float avg_dz_r = depth_image_z_dx_integral_horisontal_.at<float>(v, u + scan_line_width_right) - depth_image_z_dx_integral_horisontal_.at<float>(v, u + 1);

                // estimate angle difference
                const float alpha_left = fast_atan2f_1(-avg_dz_l, -avg_dx_l);
                const float alpha_right = fast_atan2f_1(avg_dz_r, avg_dx_r);
                const float diff = fabs(alpha_left - alpha_right);
                if (diff != 0 && (diff < (180. - min_detectable_edge_angle) * 1. / 180. * M_PI || diff > (180. + min_detectable_edge_angle) * 1. / 180. * M_PI))
                {
                    if (edge_start_index == -1)
                        edge_start_index = u;
                    const float dist = fabs(CV_PI - diff);
                    if (dist > max_edge_strength)
                    {
                        max_edge_strength = dist;
                        edge_start_index = u;
                    }
                }
                else
                {
                    if (edge_start_index != -1)
                    {
                        edge_.at<uchar>(v, edge_start_index) = 255;
                        edge_start_index = -1;
                        max_edge_strength = 0;
                    }
                }
            }
        }
    }

    void ScanLineVertical(const DepthFrame &frame, cv::Mat &distance_map_vertical, int32_t max_scan_line_width = 20, int32_t min_scan_line_width = 1, int32_t scan_line_width_init = 10, int32_t min_detectable_edge_angle = 45)
    {
        // TODO: zero copy?
        cv::Mat z_image = frame.get_z_image();
        cv::Mat depth_image_y_dy_transposed = depth_image_y_dy_.clone().t();
        cv::Mat depth_image_z_dy_transposed = depth_image_z_dy_.clone().t();

        const int max_uy = depth_image_z_dy_transposed.cols - max_scan_line_width - 2;
        const int max_vy = depth_image_z_dy_transposed.rows - max_scan_line_width - 2;

#pragma omp parallel for
        for (int v = max_scan_line_width + 1; v < max_vy; ++v)
        {
            int scan_line_width = scan_line_width_init; // width of scan line left or right of a query pixel, measured in [px]
            int last_line_width = scan_line_width;
            int edge_start_index = -1;
            float max_edge_strength = 0;
            for (int u = max_scan_line_width + 1; u < max_uy; ++u)
            {
                const float depth = z_image.at<float>(u, v);
                if (depth == 0.f)
                    continue;

                scan_line_width = std::min(int(scan_line_model_m_ * depth + scan_line_model_n_), max_scan_line_width - 1); //TODO:  max_scan_line_width - 1 is not corresponded with raw code
                if (scan_line_width <= min_scan_line_width)
                    scan_line_width = last_line_width;
                else
                    last_line_width = scan_line_width;

                int scan_line_height_upper = scan_line_width;
                int scan_line_height_lower = scan_line_width;

                if (enable_adaptive_scan_line_)
                {
                    if (AdaptScanLine(scan_line_height_upper, scan_line_height_lower, distance_map_vertical, v, u, min_scan_line_width) == false)
                    {
                        edge_start_index = -1;
                        max_edge_strength = 0.f;
                        continue;
                    }
                }
                else
                {
                    if (edge_integral_.at<int>(u + scan_line_width + 1, v + 1) - edge_integral_.at<int>(u - scan_line_width - 2, v + 1) - edge_integral_.at<int>(u + scan_line_width + 1, v) + edge_integral_.at<int>(u - scan_line_width - 2, v) != 0)
                    {
                        edge_start_index = -1;
                        max_edge_strength = 0.f;
                        continue;
                    }
                }

                const float avg_dx_l = depth_image_y_dy_integral_vertical_.at<float>(v, u - 1) - depth_image_y_dy_integral_vertical_.at<float>(v, u - scan_line_height_upper);
                const float avg_dx_r = depth_image_y_dy_integral_vertical_.at<float>(v, u + scan_line_height_lower) - depth_image_y_dy_integral_vertical_.at<float>(v, u + 1);
                const float avg_dz_l = depth_image_z_dy_integral_vertical_.at<float>(v, u - 1) - depth_image_z_dy_integral_vertical_.at<float>(v, u - scan_line_height_upper);
                const float avg_dz_r = depth_image_z_dy_integral_vertical_.at<float>(v, u + scan_line_height_lower) - depth_image_z_dy_integral_vertical_.at<float>(v, u + 1);

                const float alpha_left = fast_atan2f_1(-avg_dz_l, -avg_dx_l);
                const float alpha_right = fast_atan2f_1(avg_dz_r, avg_dx_r);
                const float diff = fabs(alpha_left - alpha_right);
                if (diff != 0 && (diff < (180. - min_detectable_edge_angle) * 1. / 180. * CV_PI || diff > (180. + min_detectable_edge_angle) * 1. / 180. * CV_PI))
                {
                    if (edge_start_index == -1)
                        edge_start_index = u;
                    const float dist = fabs(M_PI - diff);
                    if (dist > max_edge_strength)
                    {
                        max_edge_strength = dist;
                        edge_start_index = u;
                    }
                }
                else
                {
                    if (edge_start_index != -1)
                    {
                        edge_.at<uchar>(edge_start_index, v) = 255;
                        edge_start_index = -1;
                        max_edge_strength = 0;
                    }
                }
            }
        }
    }

    void FlipNormalTowardsViewpoint(const float &px, const float &py, const float &pz, const float &vp_x, const float &vp_y, const float &vp_z,
                                    float &nx, float &ny, float &nz)
    {
        float vp_x_centrized = vp_x - px;
        float vp_y_centrized = vp_y - py;
        float vp_z_centrized = vp_z - pz;

        // Dot product between the (viewpoint - point) and the plane normal
        float cos_theta = (vp_x_centrized * nx + vp_y_centrized * ny + vp_z_centrized * nz);

        // Flip the plane normal
        if (cos_theta < 0)
        {
            nx *= -1;
            ny *= -1;
            nz *= -1;
        }
    }

    void UpdateEdge(int32_t max_scan_line_width = 20)
    {
        for (int v = max_scan_line_width; v < edge_.rows - max_scan_line_width; ++v)
        {
            for (int u = max_scan_line_width; u < edge_.cols - max_scan_line_width; ++u)
            {
                if (edge_.at<uchar>(v, u) == 0)
                {
                    if (edge_.at<uchar>(v, u + 1) != 0 && edge_.at<uchar>(v + 1, u) != 0 && edge_.at<uchar>(v + 1, u + 1) == 0)
                        edge_.at<uchar>(v, u) = edge_.at<uchar>(v, u + 1);
                }
                else
                {
                    if (edge_.at<uchar>(v, u + 1) == 0 && edge_.at<uchar>(v + 1, u) == 0 && edge_.at<uchar>(v + 1, u + 1) != 0)
                        edge_.at<uchar>(v, u + 1) = edge_.at<uchar>(v + 1, u + 1);
                }
            }
        }
    }

    void ComputeSurfaceNormal(const DepthFrame &frame, int32_t max_scan_line_width = 20, int32_t min_scan_line_width = 1, int32_t scan_line_width_init = 10, int32_t min_detectable_edge_angle = 45)
    {
        // TODO: zero copy?
        cv::Mat x_image = frame.get_x_image();
        cv::Mat y_image = frame.get_y_image();
        cv::Mat z_image = frame.get_z_image();
        std::vector<float> center_point = frame.get_center_point();
        float cp_x = center_point[0];
        float cp_y = center_point[1];
        float cp_z = center_point[2];
        const int max_v = depth_image_z_dx_.rows - max_scan_line_width - 2;
        const int max_u = depth_image_z_dx_.cols - max_scan_line_width - 2;
        normal_ = cv::Mat::zeros(cv::Size(camera_parameter_.image_width, camera_parameter_.image_height), CV_32FC3);

        //#pragma omp parallel for
        for (int v = max_scan_line_width + 1; v < max_v; ++v)
        {
            int scan_line_width = scan_line_width_init;
            int last_line_width = scan_line_width;

            for (int u = max_scan_line_width + 1; u < max_u; ++u)
            {
                const int idx = v * camera_parameter_.image_width + u;
                const float depth = z_image.at<float>(v, u);
                if (depth == 0.f || edge_.at<uchar>(v, u) != 0 || v <= max_scan_line_width || v >= max_v || u <= max_scan_line_width || u >= max_u)
                {
                    normal_.at<cv::Vec3f>(v, u) = {0, 0, 0};
                    continue;
                }

                // depth dependent scan line width for slope computation (1px width per 0.10m depth)
                scan_line_width = std::min(int32_t(scan_line_model_m_ * depth + scan_line_model_n_), max_scan_line_width - 1);
                if (scan_line_width <= min_scan_line_width)
                    scan_line_width = last_line_width;
                else
                    last_line_width = scan_line_width;

                int32_t scan_line_width_left = scan_line_width;
                int32_t scan_line_width_right = scan_line_width;
                int32_t scan_line_height_upper = scan_line_width;
                int32_t scan_line_height_lower = scan_line_width;

                if (enable_adaptive_scan_line_)
                {
                    if (AdaptScanLineNormal(scan_line_width_left, scan_line_width_right, distance_map_horizontal_, u, v, min_scan_line_width) == false ||
                        AdaptScanLineNormal(scan_line_height_upper, scan_line_height_lower, distance_map_vertical_, u, v, min_scan_line_width) == false)
                    {
                        normal_.at<cv::Vec3f>(v, u) = {0, 0, 0};
                        continue;
                    }
                }
                else
                {
                    if ((edge_integral_.at<int>(v + 1, u + scan_line_width + 1) - edge_integral_.at<int>(v + 1, u - scan_line_width - 2) - edge_integral_.at<int>(v, u + scan_line_width + 1) + edge_integral_.at<int>(v, u - scan_line_width - 2) != 0) ||
                        (edge_integral_.at<int>(v + scan_line_width + 1, u + 1) - edge_integral_.at<int>(v - scan_line_width - 2, u + 1) - edge_integral_.at<int>(v + scan_line_width + 1, u) + edge_integral_.at<int>(v - scan_line_width - 2, u) != 0))
                    {
                        normal_.at<cv::Vec3f>(v, u) = {0, 0, 0};
                        continue;
                    }
                }

                const float avg_dx1 = depth_image_x_dx_integral_horisontal_.at<float>(v, u + scan_line_width_right) - depth_image_x_dx_integral_horisontal_.at<float>(v, u - scan_line_width_left);
                const float avg_dz1 = depth_image_z_dx_integral_horisontal_.at<float>(v, u + scan_line_width_right) - depth_image_z_dx_integral_horisontal_.at<float>(v, u - scan_line_width_left);
                const float avg_dy2 = depth_image_y_dy_integral_vertical_.at<float>(u, v + scan_line_height_lower) - depth_image_y_dy_integral_vertical_.at<float>(u, v - scan_line_height_upper);
                const float avg_dz2 = depth_image_z_dy_integral_vertical_.at<float>(u, v + scan_line_height_lower) - depth_image_z_dy_integral_vertical_.at<float>(u, v - scan_line_height_upper);

                const Eigen::Vector3f v1(avg_dx1, 0, avg_dz1);
                const Eigen::Vector3f v2(0, avg_dy2, avg_dz2);
                Eigen::Vector3f n = (v2.cross(v1)).normalized();

                float px = x_image.at<uint16_t>(v, u);
                float py = y_image.at<uint16_t>(v, u);
                float pz = z_image.at<uint16_t>(v, u);

                FlipNormalTowardsViewpoint(px, py, pz, cp_x, cp_y, cp_z, n(0), n(1), n(2));
                normal_.at<cv::Vec3f>(v, u)[0] = n(0);
                normal_.at<cv::Vec3f>(v, u)[1] = n(1);
                normal_.at<cv::Vec3f>(v, u)[2] = n(2); // depth dependent scan line width for slope computation (1px width per 0.10m depth)
            }
        }
    }

    cv::Mat depth_image_x_dx_, depth_image_y_dy_, depth_image_z_dx_, depth_image_z_dy_; //Spatial differences of depth image
    cv::Mat edge_image_;

    bool enable_adaptive_scan_line_;
    omp_lock_t omp_lock_;
    float scan_line_model_n_, scan_line_model_m_;

    // integral images
    cv::Mat depth_image_x_dx_integral_horisontal_, depth_image_z_dx_integral_horisontal_;
    cv::Mat depth_image_y_dy_integral_vertical_, depth_image_z_dy_integral_vertical_;

    cv::Mat distance_map_horizontal_, distance_map_vertical_;
    CameraParameter camera_parameter_;
    bool enable_edge_filter_;
    cv::Mat edge_, edge_integral_;
    cv::Mat normal_;
};