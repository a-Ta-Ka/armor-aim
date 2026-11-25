#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <array>

namespace armor_detector {

// 神经网络识别出的单块装甲板信息
struct NnArmor {
    std::array<cv::Point2f, 4> corners; // 左上、右上、右下、左下
    int color;                          // 0 蓝，1 红
    int number;                         // 模型编号类别 (0~8)
    float confidence;                   // 用于排序的综合分数
};

class NnDetector {
public:
    NnDetector() = default;
    ~NnDetector() = default;

    // 初始化：加载模型，配置参数
    void init(const std::string &model_path,
              float color_threshold,
              float number_threshold,
              float nms_threshold);

    // 识别接口：输入原图，输出装甲板列表
    std::vector<NnArmor> detect(const cv::Mat &image);

private:
    // OpenVINO对象
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    // 过滤阈值
    float color_threshold_ = 0.5f;
    float number_threshold_ = 0.5f;
    float nms_threshold_ = 0.4f;

    // 预处理函数
    cv::Mat letterbox(const cv::Mat &src, float &scale_x, float &scale_y, int &pad_x, int &pad_y);
};

} // namespace armor_detector