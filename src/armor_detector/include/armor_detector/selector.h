#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <armor_detector/DetectionFrame.h>
#include <armor_detector/TargetSelect.h>
#include <visualization_msgs/MarkerArray.h>
#include "armor_detector/armor_types.h"

namespace armor_detector
{
    class SelectorNodelet : public nodelet::Nodelet
    {
    public:
        virtual void onInit();
    private:
        void onDetection(const armor_detector::DetectionFrame::ConstPtr &msg);
        bool makeObservation(const armor_detector::Detection &d, Observation &obs) const;
        void initializeTarget(const Observation &obs, const std_msgs::Header &header, uint32_t id);
        int updateTarget(const Observation &obs, double dt);
        void updateBaseline(const Observation &obs, double t, int face,
                            cv::Point3d &velocity);
  
        cv::Point3d predictBaseline(double t, cv::Point3d &velocity);
        cv::Point3d predictArmor(double T, const cv::Point3d &p_current,
                                 const cv::Point3d &V, double theta,
                                 int face) const;
        cv::Point3d armorPosition(const cv::Point3d &C, double theta, int face) const;

        void publishTarget(const std_msgs::Header &header, bool valid, bool predicted,
                           const cv::Point3d &position,
                           const cv::Point3d &position_predict,
                           const cv::Point3d &position_baseline,
                           const cv::Point3d &car_center,
                           double theta, double omega, int face);
        void publishInvalid(const std_msgs::Header &header);
        void publishMarkers(const std_msgs::Header &header,
                            const cv::Point3d &car_center,
                            const cv::Point3d &current,
                            const cv::Point3d &predicted,
                            const cv::Point3d &baseline,
                            bool valid, bool extrapolated);

        std::string detection_topic_ = "/ft/detection";
        std::string target_topic_ = "/ft/target";
        std::string marker_topic_ = "/ft/target/marker";
        int queue_size_ = 2;

        double predict_T_ = 0.2;        // 提前量时间
        double radius_ = 0.205;         // 车体半径
        int plate_count_ = 4;           // 装甲板面数
        double lost_timeout_ = 0.5;     // 彻底丢失超时
        double dt_max_ = 0.5;           // 最大帧间时间差
        double gate_center_ = 0.6;      // 关联波门
        double face_on_min_ = 0.15;     // 最小正对程度
        double alpha_center_ = 0.10;    // 中心位置修正增益
        double beta_center_ = 0.002;    // 中心速度修正增益
        double alpha_theta_ = 0.5;      // 相位修正增益
        double beta_theta_ = 0.03;      // 角速度修正增益
        double face_hysteresis_ = 0.35; // 换面防抖滞回阈值
        double max_omega_ = 8.0;        // 最大角速度限制
        double max_speed_ = 2.0;        // 最大平移速度限制
        double n_y_alpha_ = 0.2;        // 法向垂直分量平滑系数

        double baseline_q_pos_ = 1e-4;  // 位置过程噪声
        double baseline_q_vel_ = 3e-3;  // 速度过程噪声
        double baseline_r_pos_ = 1e-3;  // 位置测量噪声

        ros::Subscriber det_sub_;
        ros::Publisher target_pub_;
        ros::Publisher marker_pub_;

        bool has_target_ = false;
        uint32_t target_id_ = 0;
        uint32_t next_id_ = 1;
        bool target_red_ = false;

        cv::Point3d C_ = cv::Point3d(0, 0, 0);  // 车体中心
        cv::Point3d V_ = cv::Point3d(0, 0, 0);  // 车体中心速度
        double theta_ = 0.0;                    // 连续相位
        double omega_ = 0.0;                    // 角速度
        double n_y_ = 0.0;                      // n_in 的垂直分量
        int face_ = 0;                          // 当前面号

        double state_stamp_ = -1.0;             // 上次滤波更新时间
        double last_hit_stamp_ = -1.0;          // 上次命中观测时间

        cv::KalmanFilter baseline_kf_; // 基础线性卡尔曼滤波器
        bool baseline_ready_ = false;  // 就位
        int baseline_face_ = -1;       // 绑定面号
        double baseline_stamp_ = -1.0; // 上次时间
    };
}
