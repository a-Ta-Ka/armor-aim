#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

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
        void updateTarget(const Observation &obs, double dt);

        // 同一可见装甲板在 t+T 时刻的预测位置，以当前板为锚点
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

        double predict_T_ = 0.2;
        double radius_ = 0.205;
        int plate_count_ = 4;
        double lost_timeout_ = 0.5;
        double dt_max_ = 0.5;
        double gate_center_ = 0.6;
        double face_on_min_ = 0.15;
        double alpha_center_ = 0.10;
        double beta_center_ = 0.002;
        double alpha_theta_ = 0.5;
        double beta_theta_ = 0.03;
        double face_hysteresis_ = 0.35;
        double max_omega_ = 8.0;
        double max_speed_ = 2.0;
        double n_y_alpha_ = 0.2;

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

        // 基础 position+velocity*T 对比：用装甲板中心有限差分速度。
        bool has_last_p_ = false;
        cv::Point3d last_p_obs_ = cv::Point3d(0, 0, 0);
        double last_p_stamp_ = -1.0;
        cv::Point3d v_obs_ = cv::Point3d(0, 0, 0);
    };
}
