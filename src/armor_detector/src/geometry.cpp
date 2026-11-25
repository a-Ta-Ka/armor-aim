#include "armor_detector/geometry.h"

#include <ros/ros.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace armor_detector
{
    float quadWidth(const cv::Point2f q[4]) 
    { return (cv::norm(q[1] - q[0]) + cv::norm(q[2] - q[3])) * 0.5f;}
    float quadHeight(const cv::Point2f q[4])
    { return (cv::norm(q[3] - q[0]) + cv::norm(q[2] - q[1])) * 0.5f;}
    cv::Point2f quadCenter(const cv::Point2f q[4])
    { return (q[0] + q[1] + q[2] + q[3]) * 0.25f;}
    
    bool estimatePose(const cv::Point2f quad[4],
                      double armor_w, double armor_l,
                      const cv::Mat &K, const cv::Mat &D,
                      cv::Mat &rvec, cv::Mat &tvec)
    {
        const double half_w = armor_w * 0.5;
        const double half_h = armor_l * 0.5;
        const std::vector<cv::Point3f> obj =
        {
            {(float)-half_w, (float)-half_h, 0.0f}, // 左上
            {(float)half_w, (float)-half_h, 0.0f},  // 右上
            {(float)half_w, (float)half_h, 0.0f},   // 右下
            {(float)-half_w, (float)half_h, 0.0f},  // 左下
        };
        const std::vector<cv::Point2f> img(quad, quad + 4);

        try
        {
            const bool solved = cv::solvePnP(
                obj, img, K, D, rvec, tvec,
                false, cv::SOLVEPNP_ITERATIVE);

            if (!solved)
                return false;

            // 排除 NaN、Inf，避免污染 TF 和后续滤波。
            if (!cv::checkRange(rvec) || !cv::checkRange(tvec))
                return false;

            // 模型原点是装甲板中心，中心应位于相机前方。
            if (tvec.at<double>(2) <= 0.0)
                return false;

            return true;
        }
        catch (const cv::Exception &e)
        {
            ROS_WARN_THROTTLE(1.0, "PnP failed: %s", e.what());
            return false;
        }
    }//pnp

    tf2::Quaternion toQuaternion(const cv::Mat &rvec)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        tf2::Matrix3x3 m(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                         R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                         R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
        tf2::Quaternion q;
        m.getRotation(q);
        return q;
    }//旋转向量转四元数

    void drawArmor(cv::Mat &img,
                   const cv::Point2f quad[4],
                   const cv::Point2f &center,
                   const cv::Scalar &center_color,
                   bool pose_ok,
                   const cv::Mat &rvec, const cv::Mat &tvec,
                   const cv::Mat &K, const cv::Mat &D)
    {
        for (int m = 0; m < 4; m++)
        {
            cv::line(img, quad[m], quad[(m + 1) % 4], cv::Scalar(255, 0, 255), 2);
            cv::circle(img, quad[m], 3, cv::Scalar(0, 255, 0), -1);
        }

        if (pose_ok)
        {
            std::vector<cv::Point3f> axis = {{0, 0, 0}, {0.05f, 0, 0}, {0, 0.05f, 0}, {0, 0, 0.05f}};
            std::vector<cv::Point2f> proj;
            cv::projectPoints(axis, rvec, tvec, K, D, proj);
            cv::circle(img, center, 5, center_color, -1);
            cv::line(img, proj[0], proj[1], cv::Scalar(0, 0, 255), 2);
            cv::line(img, proj[0], proj[2], cv::Scalar(0, 255, 0), 2);
            cv::line(img, proj[0], proj[3], cv::Scalar(255, 0, 0), 2);
        }
    }//画装甲板

} // namespace armor_detector
