#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace armor_detector
{

    class DetectorNodelet : public nodelet::Nodelet
    {
    public:
        virtual void onInit();

    private:
        void imageCallback(const sensor_msgs::ImageConstPtr &msg);

        void getcontours(const cv::Mat &g_mask, cv::Mat &g_img, bool is_red_armor);

        void update(cv::Mat &g_img, cv::Mat &mask);

        int BH_MIN_ = 80, BH_MAX_ = 120, RH_MIN_ = 0, RH_MAX_ = 40;
        int RH2_MIN_ = 170, RH2_MAX_ = 179;
        int S_MIN_ = 80, S_MAX_ = 255, V_MIN_ = 160, V_MAX_ = 255;
        int MIN_AREA_ = 10;
        cv::Mat kernel_;

        ros::Publisher debug_pub_;
        ros::Subscriber image_sub_;
    };

} // namespace armor_detector