#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "armor_detector/armor_types.h"

namespace armor_detector
{
    class TvDetector
    {
    public:
        struct Params
        {
            BarFilter bar;        // 灯条筛选
            PairParam pair;       // 灯条配对
            MirrorParam mirror;   // 倒影判据
            HsvRange hsv;         // 六个 HSV 区间
        };
        TvDetector();
        void setParams(const Params &p) { params_ = p; }
        void find(const cv::Mat &bgr, cv::Mat &canvas, std::vector<Armor> &out);

    private:
        struct Pair
        {
            int i, j;
            float cost;
            bool large;
        };
        void maskForColor(const cv::Mat &hsv,
                          const cv::Scalar &lo, const cv::Scalar &hi,
                          const cv::Mat &mask_extra, cv::Mat &mask);
        void detectInMask(const cv::Mat &mask, bool is_red,
                          cv::Mat &canvas, std::vector<Armor> &out);
        void extractBars(const cv::Mat &mask, std::vector<cv::RotatedRect> &bars) const;
        bool hasBarBetween(const std::vector<cv::RotatedRect> &bars, int i, int j) const;
        void pairBars(const std::vector<cv::RotatedRect> &bars, std::vector<Pair> &pairs) const;
        void buildQuads(const std::vector<cv::RotatedRect> &bars,
                        const std::vector<Pair> &pairs,
                        std::vector<Armor> &quads) const;
        void markReflections(std::vector<Armor> &quads) const;
        bool isFloorReflection(const Armor &upper, const Armor &lower) const;
        void drawReflection(cv::Mat &canvas, const Armor &armor) const;

        Params params_;
        cv::Mat kernel_; 
    };
} // namespace armor_detector
