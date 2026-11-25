#pragma once

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cstddef>

#include <std_msgs/Header.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <armor_detector/DetectionFrame.h>
#include <dynamic_reconfigure/server.h>
#include <armor_detector/ArmorDetectorConfig.h>

#include "armor_detector/armor_types.h"
#include "armor_detector/tv_detector.h"
#include "armor_detector/nn_detector.h"

namespace armor_detector
{
    typedef armor_detector::ArmorDetectorConfig Config;

    class DetectorNodelet : public nodelet::Nodelet
    {
    public:
        virtual void onInit();
    private:
        void loadParams();                // 读 YAML、建动态参数服务、按需加载模型
        void setupInterfaces();           // 订阅与发布

        // 回调
        void onCameraInfo(const sensor_msgs::CameraInfoConstPtr &msg);
        void onImage(const sensor_msgs::ImageConstPtr &msg);
        void onReconfigure(Config &config, uint32_t level);

        // 前端 → 统一后处理
        void findWithTv(const cv::Mat &image, cv::Mat &canvas);
        void findWithNn(const cv::Mat &image, cv::Mat &canvas);
        bool appendArmor(const Armor &armor, cv::Mat &canvas);

        // appendArmor 内部四件事
        bool estimatePose(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec) const;
        void makeDetection(const Armor &armor, int id,
                           const cv::Mat &tvec, const tf2::Quaternion &q);
        void broadcastTf(int id, const cv::Mat &tvec, const tf2::Quaternion &q);

        boost::shared_ptr<dynamic_reconfigure::Server<Config>> reconf_server_;

        TvDetector tv_detector_;
        NnDetector nn_detector_;
        TvDetector::Params tv_params_; 

        int det_id_ = 0;

        // 装甲板物理尺寸（米）
        double wl_large_ = 4.0, wl_small_ = 2.4;
        double armor_large_w_ = 0.2256;
        double armor_large_l_ = 0.056;
        double armor_small_w_ = 0.135;
        double armor_small_l_ = 0.056;

        // 神经网络
        float nn_color_threshold_ = 0.5f;
        float nn_number_threshold_ = 0.5f;
        float nn_nms_threshold_ = 0.4f;
        std::string detection_mode_ = "tv";
        std::string nn_model_path_;
        std::size_t nn_frames_ = 0;
        std::size_t nn_timed_frames_ = 0;
        double nn_total_ms_ = 0.0;

        cv::Mat K_, D_;

        std::string camera_frame_;
        std_msgs::Header cur_header_; // 当前帧图像 header
        std::vector<armor_detector::Detection> dets_;
        tf2_ros::TransformBroadcaster tf_broadcaster_;

        ros::Publisher debug_pub_;
        ros::Publisher detection_pub_;
        image_transport::Subscriber image_sub_;
        ros::Subscriber info_sub_;
    };
} // namespace armor_detector
