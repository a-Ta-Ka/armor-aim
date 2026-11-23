#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <string>

#include <armor_detector/DetectionFrame.h>
#include <opencv2/opencv.hpp>
#include <armor_detector/TrackTarget.h>
#include <visualization_msgs/Marker.h>

namespace armor_detector
{
    class TrackerNodelet : public nodelet::Nodelet
    {
    public:
        virtual void onInit();

    private:
        void detectionCallback(const armor_detector::DetectionFrame::ConstPtr &msg);
        void resetKalman(const cv::Point3d &p);
        void publishTrack(const std_msgs::Header &header, bool tracking);
        void publishMarker(const std_msgs::Header &header, bool tracking, const geometry_msgs::Point &p);

        std::string detection_topic_= "/ft/detection";
        std::string track_topic_ = "/ft/track";
        std::string marker_topic_ = "/ft/debug/marker";
        ros::Subscriber det_sub_;
        ros::Publisher track_pub_;
        ros::Publisher marker_pub_;
        int queue_size_ = 2;

        cv::KalmanFilter kf_;
        int track_id_ = 0;         // 轨迹编号
        bool has_track_ = false;   // 当前有没有轨迹
        double last_stamp_ = -1.0; // 上一帧图像时间戳
        double last_hit_stamp_ = -1.0; // 上一次有效观测命中的时间

        double predict_t_ = 0.1;    // 提前量，秒
        double max_match_dist_ = 0.5; // 关联门限 m
        double lost_timeout_ = 0.5;   // 连续丢检超过它就删轨迹 s
        double dt_max_ = 0.5;         // 帧间隔超过它认为是断流 s
        double q_pos_ = 1e-4;         // 位置过程噪声 m^2
        double q_vel_ = 0.003;         // 速度过程噪声 (m/s)^2
        double r_pos_ = 1e-3;         // 观测噪声 m^2
        double init_pos_var_ = 1.0;   // 初始位置方差 m^2
        double init_vel_var_ = 10.0;  // 初始速度方差 (m/s)^2
    };
}