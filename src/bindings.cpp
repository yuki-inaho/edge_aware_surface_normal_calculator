#include <string>
#include "pybind11/pybind11.h"
#include "ndarray_converter.h"
#include "surface_normal_calculator.h"
#include "depth_frame.h"
#include "struct.h"

namespace py = pybind11;

PYBIND11_MODULE(edge_aware_surface_normal_calculator, m)
{
    NDArrayConverter::init_numpy();
    py::class_<CameraParameter>(m, "CameraParameter")
        .def(
            py::init<float, float, float, float, int32_t, int32_t>(),
            py::arg("fx"),
            py::arg("fy"),
            py::arg("cx"),
            py::arg("cy"),
            py::arg("image_width"),
            py::arg("image_height"));

    py::class_<DepthFrame>(m, "DepthFrame")
        .def(
            py::init<const cv::Mat &, const CameraParameter &>(),
            py::arg("depth_image"),
            py::arg("camera_parameter"));

    py::class_<SurfaceNormalCalculator>(m, "SurfaceNormalCalculator")
        .def(
            py::init<CameraParameter, bool, bool, int32_t>(),
            py::arg("camera_parameter"),
            py::arg("enable_adaptive_scan_line") = true,
            py::arg("enable_edge_filter") = true,
            py::arg("scan_line_width_at_2m") = 15)
        .def("compute", &SurfaceNormalCalculator::compute)
        .def("get_edge_image", &SurfaceNormalCalculator::get_edge_image)
        .def("get_surface_normal_image", &SurfaceNormalCalculator::get_surface_normal_image);
}