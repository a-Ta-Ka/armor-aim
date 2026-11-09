#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

namespace armor_detector
{

    class CameraNodelet : public nodelet::Nodelet
    {
    public:
        virtual void onInit();

    private:
        void imageCallback(const sensor_msgs::ImageConstPtr &msg);
        void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg);

        image_transport::Publisher debug_pub_;
        ros::Subscriber info_sub_;
        image_transport::Subscriber image_sub_;

        cv::Mat K_, D_;
        int img_width_ = 0;
        int img_height_ = 0;
        bool has_info_ = false;
    };

} // namespace armor_detector