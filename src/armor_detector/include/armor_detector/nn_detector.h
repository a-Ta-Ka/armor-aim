#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "armor_detector/armor_types.h"

namespace armor_detector
{
    class NnDetector
    {
    public:
        void init(const std::string &model_path,
                  float color_threshold,
                  float number_threshold,
                  float nms_threshold);
        void setLargeRatio(float ratio) { large_ratio_ = ratio; }//大小判定
        std::vector<Armor> find(const cv::Mat &image);

    private:
        struct Candidate
        {
            Armor armor;
            cv::Rect box;
            float score = 0.f;
        }; //候选

        //预处理
        void preprocess(const cv::Mat &image, cv::Mat &rgb_float,
                        float &scale_x, float &scale_y, int &pad_x, int &pad_y) const;
        //一次推理
        void infer(const cv::Mat &rgb_float);
        //后处理
        void decode(const float *out, float scale_x, float scale_y,
                    int pad_x, int pad_y, std::vector<Candidate> &cands) const;
        //去重
        void nms(std::vector<Candidate> &cands, std::vector<Armor> &out) const;
        //处理集
        cv::Mat letterbox(const cv::Mat &src, float &scale_x, float &scale_y,
                          int &pad_x, int &pad_y) const;

        ov::Core core_;
        ov::CompiledModel compiled_model_;
        ov::InferRequest infer_request_;

        float color_threshold_ = 0.5f;
        float number_threshold_ = 0.5f;
        float nms_threshold_ = 0.4f;
        float large_ratio_ = 2.9f;
    };

} // namespace armor_detector
