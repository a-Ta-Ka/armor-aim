#include "armor_detector/nn_detector.h"
#include <iostream>
#include <algorithm>

#include <stdexcept>
#include <cmath>
#include <opencv2/dnn.hpp>

namespace armor_detector {

void NnDetector::init(const std::string &model_path, float color_threshold, float number_threshold, float nms_threshold) {
    for (const float threshold : {color_threshold, number_threshold, nms_threshold})
    {
        if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f)
            throw std::invalid_argument("NN thresholds must be finite and in [0, 1].");
    }

    color_threshold_ = color_threshold;
    number_threshold_ = number_threshold;
    nms_threshold_ = nms_threshold;

    // 1. 读取模型
    auto model = core_.read_model(model_path);
    
    // 2. 编译模型 (使用 CPU)
    compiled_model_ = core_.compile_model(model, "CPU");
    
    // 3. 创建推理请求
    infer_request_ = compiled_model_.create_infer_request();

    const auto input = infer_request_.get_input_tensor();
    if (input.get_element_type() != ov::element::f32 || input.get_shape() != ov::Shape{1, 3, 480, 640})
    {
        throw std::runtime_error("Unexpected NN input type or shape.");
    }
}

// 预处理：等比缩放，居中补边值 124
cv::Mat NnDetector::letterbox(const cv::Mat &src, float &scale_x, float &scale_y, int &pad_x, int &pad_y) {
    int target_w = 640;
    int target_h = 480;
    
    // 计算缩放比例
    const float scale = std::min(target_w / (float)src.cols, target_h / (float)src.rows);
    const int new_w = std::max(1, std::min(target_w, static_cast<int>(std::round(src.cols * scale))));
    const int new_h = std::max(1, std::min(target_h, static_cast<int>(std::round(src.rows * scale))));
    scale_x = static_cast<float>(new_w) / src.cols;
    scale_y = static_cast<float>(new_h) / src.rows;
    
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    
    // 计算补边
    pad_x = (target_w - new_w) / 2;
    pad_y = (target_h - new_h) / 2;
    
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_y, target_h - new_h - pad_y, 
                       pad_x, target_w - new_w - pad_x, cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));
    
    return padded;
}

std::vector<NnArmor> NnDetector::detect(const cv::Mat &image) {
    if (image.empty() || image.type() != CV_8UC3)
        throw std::runtime_error("NN input must be a non-empty BGR image.");

    std::vector<NnArmor> results;

    // 1. 预处理
    float scale_x, scale_y;
    int pad_x, pad_y;
    cv::Mat padded = letterbox(image, scale_x, scale_y, pad_x, pad_y);

    // 2. 转换为 OpenVINO 张量 (BGR -> RGB, 归一化, HWC -> CHW)
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    ov::Tensor input_tensor = infer_request_.get_input_tensor();
    float* input_data = input_tensor.data<float>();
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 480; ++h) {
            for (int w = 0; w < 640; ++w) {
                input_data[c * 480 * 640 + h * 640 + w] = rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    infer_request_.set_input_tensor(input_tensor);

    // 3. 执行推理
    infer_request_.infer();

    // 4. 解析输出 [1, 21, 6300]
    ov::Tensor output_tensor = infer_request_.get_output_tensor();

    if (output_tensor.get_element_type() != ov::element::f32 ||
        output_tensor.get_shape() != ov::Shape{1, 21, 6300})
    {
        throw std::runtime_error("Unexpected NN output type or shape.");
    }

    cv::Mat values(21, 6300, CV_32F, output_tensor.data<float>());
    if (!cv::checkRange(values))
        throw std::runtime_error("NN output contains NaN or Inf.");

    const float *out_data = output_tensor.data<float>();
    const int num_candidates = 6300;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<NnArmor> candidates;

    for (int i = 0; i < num_candidates; ++i)
    {
        int best_color = 0;
        float max_color_score = out_data[i];

        for (int c = 1; c < 4; ++c)
        {
            const float score = out_data[c * num_candidates + i];
            if (score > max_color_score)
            {
                max_color_score = score;
                best_color = c;
            }
        }

        if (best_color > 1 || max_color_score < color_threshold_)
            continue;
        // 找数字最大值 (索引 4~12)
        int best_number = -1;
        float max_num_score = 0;
        for (int n = 4; n <= 12; ++n) {
            float score = out_data[n * num_candidates + i];
            if (score > max_num_score) {
                max_num_score = score;
                best_number = n - 4;
            }
        }
        if (max_num_score < number_threshold_) continue;

        // 提取四个角点 (13~20) -> 模型顺序：左上、左下、右下、右上
        std::vector<cv::Point2f> pts(4);
        for (int k = 0; k < 4; ++k) {
            float x = out_data[(13 + k * 2) * num_candidates + i];
            float y = out_data[(14 + k * 2) * num_candidates + i];
            
            // 逆映射回原图
            pts[k].x = (x - pad_x) / scale_x;
            pts[k].y = (y - pad_y) / scale_y;
        }

        // 重排角点：左上、右上、右下、左下
        std::array<cv::Point2f, 4> corners;
        corners[0] = pts[0]; // 左上
        corners[1] = pts[3]; // 右上
        corners[2] = pts[2]; // 右下
        corners[3] = pts[1]; // 左下

        const std::vector<cv::Point2f> polygon(
            corners.begin(), corners.end());

        if (!cv::isContourConvex(polygon) ||
            std::abs(cv::contourArea(polygon)) < 1.0)
        {
            continue;
        }

        cv::Rect box = cv::boundingRect(std::vector<cv::Point2f>(corners.begin(), corners.end()));

        NnArmor armor;
        armor.corners = corners;
        armor.color = best_color;
        armor.number = best_number;
        armor.confidence = max_color_score * max_num_score;

        boxes.push_back(box);
        confidences.push_back(armor.confidence);
        candidates.push_back(armor);
    }

    // 5. NMS 去重
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.0f, nms_threshold_, indices);

    for (int idx : indices) {
        results.push_back(candidates[idx]);
    }

    return results;
}

} // namespace armor_detector