#pragma once

#include <opencv2/core.hpp>
#include <tf2/LinearMath/Quaternion.h>

namespace armor_detector
{

    float quadWidth(const cv::Point2f q[4]);// 上下边平均长度
    float quadHeight(const cv::Point2f q[4]);// 左右边平均长度

    cv::Point2f quadCenter(const cv::Point2f q[4]);// 四角点平均
    bool estimatePose(const cv::Point2f quad[4],
                      double armor_w, double armor_l,
                      const cv::Mat &K, const cv::Mat &D,
                      cv::Mat &rvec, cv::Mat &tvec);//PNP

    tf2::Quaternion toQuaternion(const cv::Mat &rvec);// 旋转向量 → 四元数
    void drawArmor(cv::Mat &img,
                   const cv::Point2f quad[4],
                   const cv::Point2f &center,
                   const cv::Scalar &center_color,
                   bool pose_ok,
                   const cv::Mat &rvec, const cv::Mat &tvec,
                   const cv::Mat &K, const cv::Mat &D);// 画装甲板

} // namespace armor_detector
