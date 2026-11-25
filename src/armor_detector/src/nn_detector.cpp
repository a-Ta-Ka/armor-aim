#include "armor_detector/nn_detector.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/dnn.hpp>

#include "armor_detector/geometry.h"

namespace armor_detector
{
    void NnDetector::init(const std::string &model_path,
                          float color_threshold,
                          float number_threshold,
                          float nms_threshold)
    {
        for (const float threshold : {color_threshold, number_threshold, nms_threshold})
        {
            if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f)
                throw std::invalid_argument("NN thresholds must be finite and in [0, 1].");
        }
        color_threshold_ = color_threshold;
        number_threshold_ = number_threshold;
        nms_threshold_ = nms_threshold;

        //读取模型
        auto model = core_.read_model(model_path);
        //编译模型
        compiled_model_ = core_.compile_model(model, "CPU");
        //创建推理请求
        infer_request_ = compiled_model_.create_infer_request();
        const auto input = infer_request_.get_input_tensor();
        if (input.get_element_type() != ov::element::f32 ||
            input.get_shape() != ov::Shape{1, 3, 480, 640})
        {
            throw std::runtime_error("Unexpected NN input type or shape.");
        }
    }//启动一下
    cv::Mat NnDetector::letterbox(const cv::Mat &src, float &scale_x, float &scale_y,
                                  int &pad_x, int &pad_y) const
    {
        const int target_w = 640;
        const int target_h = 480;

        const float scale = std::min(target_w / (float)src.cols, target_h / (float)src.rows);
        const int new_w = std::max(1, std::min(target_w, static_cast<int>(std::round(src.cols * scale))));
        const int new_h = std::max(1, std::min(target_h, static_cast<int>(std::round(src.rows * scale))));
        scale_x = static_cast<float>(new_w) / src.cols;
        scale_y = static_cast<float>(new_h) / src.rows;

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(new_w, new_h));

        pad_x = (target_w - new_w) / 2;
        pad_y = (target_h - new_h) / 2;

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, pad_y, target_h - new_h - pad_y,
                           pad_x, target_w - new_w - pad_x,
                           cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));
        return padded;
    }//缩放补边
    void NnDetector::preprocess(const cv::Mat &image, cv::Mat &rgb_float,
                                float &scale_x, float &scale_y, int &pad_x, int &pad_y) const
    {
        cv::Mat padded = letterbox(image, scale_x, scale_y, pad_x, pad_y);
        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);
    }//色彩归一
    void NnDetector::infer(const cv::Mat &rgb_float)
    {
        ov::Tensor input_tensor = infer_request_.get_input_tensor();
        float *input_data = input_tensor.data<float>();
        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < 480; ++h)
            {
                for (int w = 0; w < 640; ++w)
                {
                    input_data[c * 480 * 640 + h * 640 + w] =
                        rgb_float.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        infer_request_.set_input_tensor(input_tensor);
        infer_request_.infer();
    }//推送处理
    void NnDetector::decode(const float *out, float scale_x, float scale_y,
                            int pad_x, int pad_y, std::vector<Candidate> &cands) const
    {
        const int num_candidates = 6300;

        for (int i = 0; i < num_candidates; ++i)
        {
            // 颜色：0 蓝、1 红、2 未点亮、3 其它
            int best_color = 0;
            float max_color_score = out[i];
            for (int c = 1; c < 4; ++c)
            {
                const float score = out[c * num_candidates + i];
                if (score > max_color_score)
                {
                    max_color_score = score;
                    best_color = c;
                }
            }
            if (best_color > 1 || max_color_score < color_threshold_)
                continue;

            // 编号：索引 4~12
            int best_number = -1;
            float max_num_score = 0;
            for (int n = 4; n <= 12; ++n)
            {
                const float score = out[n * num_candidates + i];
                if (score > max_num_score)
                {
                    max_num_score = score;
                    best_number = n - 4;
                }
            }
            if (max_num_score < number_threshold_)
                continue;

            // 四角点索引 13~20，模型顺序：左上、左下、右下、右上
            std::vector<cv::Point2f> pts(4);
            for (int k = 0; k < 4; ++k)
            {
                const float x = out[(13 + k * 2) * num_candidates + i];
                const float y = out[(14 + k * 2) * num_candidates + i];
                pts[k].x = (x - pad_x) / scale_x; // 逆映射回原图
                pts[k].y = (y - pad_y) / scale_y;
            }

            // 重排成 左上、右上、右下、左下
            Armor armor;
            armor.q[0] = pts[0];
            armor.q[1] = pts[3];
            armor.q[2] = pts[2];
            armor.q[3] = pts[1];

            const std::vector<cv::Point2f> polygon(armor.q, armor.q + 4);
            if (!cv::isContourConvex(polygon) ||
                std::abs(cv::contourArea(polygon)) < 1.0)
                continue;

            armor.center = quadCenter(armor.q);
            armor.width = quadWidth(armor.q);
            armor.height = quadHeight(armor.q);

            // 大小：灯条中心距 / 灯条长度
            const cv::Point2f left_center = (armor.q[0] + armor.q[3]) * 0.5f;
            const cv::Point2f right_center = (armor.q[1] + armor.q[2]) * 0.5f;
            const float left_len = cv::norm(armor.q[0] - armor.q[3]);
            const float right_len = cv::norm(armor.q[1] - armor.q[2]);
            const float avg_len = (left_len + right_len) * 0.5f;
            if (avg_len < 1e-6f)
                continue;
            const float ratio = std::abs(right_center.x - left_center.x) / avg_len;
            armor.is_large = ratio >= large_ratio_;

            armor.is_red = (best_color == 1);
            armor.number = best_number;
            armor.confidence = max_color_score * max_num_score;

            Candidate cand;
            cand.armor = armor;
            cand.box = cv::boundingRect(polygon);
            cand.score = armor.confidence;
            cands.push_back(cand);
        }
    }//整理输出
    void NnDetector::nms(std::vector<Candidate> &cands, std::vector<Armor> &out) const
    {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        boxes.reserve(cands.size());
        scores.reserve(cands.size());
        for (const Candidate &c : cands)
        {
            boxes.push_back(c.box);
            scores.push_back(c.score);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.0f, nms_threshold_, indices);

        out.clear();
        out.reserve(indices.size());
        for (int idx : indices)
            out.push_back(cands[idx].armor);
    }//去重



    std::vector<Armor> NnDetector::find(const cv::Mat &image)
    {
        if (image.empty() || image.type() != CV_8UC3)
            throw std::runtime_error("NN input must be a non-empty BGR image.");

        // 1. 预处理 + 推理
        cv::Mat rgb_float;
        float scale_x = 1.f, scale_y = 1.f;
        int pad_x = 0, pad_y = 0;
        preprocess(image, rgb_float, scale_x, scale_y, pad_x, pad_y);
        infer(rgb_float);

        // 2. 取输出并检查形状
        ov::Tensor output_tensor = infer_request_.get_output_tensor();
        if (output_tensor.get_element_type() != ov::element::f32 ||
            output_tensor.get_shape() != ov::Shape{1, 21, 6300})
        {
            throw std::runtime_error("Unexpected NN output type or shape.");
        }
        cv::Mat values(21, 6300, CV_32F, output_tensor.data<float>());
        if (!cv::checkRange(values))
            throw std::runtime_error("NN output contains NaN or Inf.");
   
            // 3. 解码 + 去重
        std::vector<Candidate> cands;
        decode(output_tensor.data<float>(), scale_x, scale_y, pad_x, pad_y, cands);

        std::vector<Armor> results;
        nms(cands, results);
        return results;
    }//一条链了

} // namespace armor_detector
