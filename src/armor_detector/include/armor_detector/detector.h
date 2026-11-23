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
#include <cmath>

#include <std_msgs/Header.h>                        //ROS消息通用头部
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>               //将ros33矩阵变为tf33矩阵
#include <tf2/LinearMath/Quaternion.h>              //变换为超强四元数
#include <tf2_ros/transform_broadcaster.h>

#include <armor_detector/DetectionFrame.h>
#include <dynamic_reconfigure/server.h>
#include <armor_detector/ArmorDetectorConfig.h>

namespace armor_detector
{
    typedef armor_detector::ArmorDetectorConfig Config;
    struct BarFilter
    {
        int min_area = 25;         // 轮廓面积下限
        int min_height = 6;       // 灯条高度下限（像素）
        float min_hw = 2.0f;       // 长宽比下限 h/w
        float max_hw = 30.0f;      // 长宽比上限
        float max_area = 20000.0f; // 面积上限（排除装甲板底色块）
        float min_fill = 0.1f;     // 填充率
    };

    struct PairParam
    {
        float dy_ratio = 0.25f;     // |dy| <= dy_ratio * max_len
        float h_diff_max = 0.15f;   // |Δh| / 平均灯条高
        float w_ratio_max = 2.5f;   // 宽灯条/窄灯条 上限
        float ang_diff_max = 8.0f;  // 角度差（度）
        float wl_min = 1.2f;        // 中心距 / 平均灯条长  下限
        float wl_max = 6.0f;        // 中心距 / 平均灯条长  上限
        float large_ratio = 2.9f;  // >= 该值判为大装甲板
        float endpoint_dy_ratio = 0.35f; // 上下端点允许的纵向差 / 平均灯条长度
    };

    class DetectorNodelet : public nodelet::Nodelet
    {
    public:
        virtual void onInit();

    private:
        void imageCallback(const sensor_msgs::ImageConstPtr &msg);
        void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg);
        void detectArmors(const cv::Mat &g_mask, cv::Mat &g_img, bool is_red_armor);
        void update(cv::Mat &g_img);
        void reconfigureCallback(Config &config, uint32_t level);
        void processColor(const cv::Mat &hsv,
                          const cv::Scalar &lo, const cv::Scalar &hi,
                          cv::Mat &g_img, bool is_red_armor,
                          const cv::Mat &mask_extra = cv::Mat());

        boost::shared_ptr<dynamic_reconfigure::Server<Config>> reconf_server_;
        BarFilter bar_filter_;
        PairParam pair_param_;

        int BH_MIN_ = 80, BH_MAX_ = 120, RH_MIN_ = 0, RH_MAX_ = 40;
        int RH2_MIN_ = 170, RH2_MAX_ = 179;
        int S_MIN_ = 80, S_MAX_ = 255, V_MIN_ = 160, V_MAX_ = 255;
        int det_id_ = 0;

        uint32_t camera_width_ = 0;
        uint32_t camera_height_ = 0;

        // 装甲板
        double wl_large_ = 4.0, wl_small_ = 2.4;
        double armor_large_w_ = 0.2256; 
        double armor_large_l_ = 0.056;   
        double armor_small_w_ = 0.135; 
        double armor_small_l_ = 0.056;  

        cv::Mat kernel_;
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
}