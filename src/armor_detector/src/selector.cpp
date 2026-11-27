#include "armor_detector/selector.h"

#include <pluginlib/class_list_macros.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace armor_detector
{
    namespace
    {
        double wrapAngle(double a)
        {
            return std::atan2(std::sin(a), std::cos(a));
        } // 约定 (-π, π]
        void clampNorm(cv::Point3d &v, double max_norm)
        {
            const double n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            if (n > max_norm && n > 1e-9)
                v *= (max_norm / n);
        }//限制向量模长
        geometry_msgs::Point toPoint(const cv::Point3d &p)
        {
            geometry_msgs::Point o;
            o.x = p.x;
            o.y = p.y;
            o.z = p.z;
            return o;
        }//转换
    }

    void SelectorNodelet::onInit()
    {
        ros::NodeHandle &n = getPrivateNodeHandle();
        n.param<int>("queue_size", queue_size_, queue_size_);
        n.param<std::string>("detection_topic", detection_topic_, detection_topic_);
        n.param<std::string>("target_topic", target_topic_, target_topic_);
        n.param<std::string>("marker_topic", marker_topic_, marker_topic_);

        n.param<double>("predict_T", predict_T_, predict_T_);
        n.param<double>("radius", radius_, radius_);
        n.param<int>("plate_count", plate_count_, plate_count_);
        n.param<double>("lost_timeout", lost_timeout_, lost_timeout_);
        n.param<double>("dt_max", dt_max_, dt_max_);
        n.param<double>("gate_center", gate_center_, gate_center_);
        n.param<double>("face_on_min", face_on_min_, face_on_min_);
        n.param<double>("alpha_center", alpha_center_, alpha_center_);
        n.param<double>("beta_center", beta_center_, beta_center_);
        n.param<double>("alpha_theta", alpha_theta_, alpha_theta_);
        n.param<double>("beta_theta", beta_theta_, beta_theta_);
        n.param<double>("face_hysteresis", face_hysteresis_, face_hysteresis_);
        n.param<double>("max_omega", max_omega_, max_omega_);
        n.param<double>("max_speed", max_speed_, max_speed_);
        n.param<double>("n_y_alpha", n_y_alpha_, n_y_alpha_);

        det_sub_ = n.subscribe(detection_topic_, queue_size_, &SelectorNodelet::onDetection, this);
        target_pub_ = n.advertise<armor_detector::TargetSelect>(target_topic_, queue_size_);
        marker_pub_ = n.advertise<visualization_msgs::MarkerArray>(marker_topic_, queue_size_);

        ROS_INFO("selector: radius=%.3f m plate_count=%d T=%.3f s",
                 radius_, plate_count_, predict_T_);
    }
    bool SelectorNodelet::makeObservation(const armor_detector::Detection &d, Observation &obs) const
    // 转换计算小车中心
    {
        const auto &p = d.pose.position;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            return false;
        if (p.z <= 0.1)
            return false;

        const auto &q = d.pose.orientation;
        const double qn = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if (qn < 1e-12)
            return false;

        cv::Point3d n;
        n.x = 2.0 * (q.x * q.z + q.y * q.w) / qn;
        n.y = 2.0 * (q.y * q.z - q.x * q.w) / qn;
        n.z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y) / qn;
        const double nn = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (nn < 1e-9)
            return false;
        n *= (1.0 / nn);

        obs.p = cv::Point3d(p.x, p.y, p.z);
        obs.n = n;
        obs.c = obs.p + radius_ * n;

        const double pn = std::sqrt(obs.p.x * obs.p.x + obs.p.y * obs.p.y + obs.p.z * obs.p.z);
        if (pn < 1e-6)
            return false;
        obs.face_on = (n.x * obs.p.x + n.y * obs.p.y + n.z * obs.p.z) / pn;

        const double h = std::sqrt(n.x * n.x + n.z * n.z);
        if (h < 1e-6)
            return false;

        obs.alpha = std::atan2(-n.z, -n.x);
        obs.det_id = static_cast<int>(d.id);
        obs.red = d.is_red;
        return true;
    }
    void SelectorNodelet::initializeTarget(const Observation &obs, const std_msgs::Header &header, uint32_t id) 
    // 初始化跟踪
    {
        has_target_ = true;
        target_id_ = id;
        target_red_ = obs.red;

        C_ = obs.c;
        V_ = cv::Point3d(0, 0, 0);
        theta_ = obs.alpha;
        omega_ = 0.0;
        n_y_ = obs.n.y;
        face_ = 0;

        state_stamp_ = header.stamp.toSec();
        last_hit_stamp_ = state_stamp_;

        has_last_p_ = true;
        last_p_obs_ = obs.p;
        last_p_stamp_ = state_stamp_;
        v_obs_ = cv::Point3d(0, 0, 0);

        ROS_INFO("selector: target %u init red=%d p=(%.2f,%.2f,%.2f) "
                 "C=(%.2f,%.2f,%.2f) alpha=%.1f deg",
                 target_id_, static_cast<int>(target_red_),
                 obs.p.x, obs.p.y, obs.p.z,
                 C_.x, C_.y, C_.z, theta_ * 180.0 / M_PI);
    }
    void SelectorNodelet::updateTarget(const Observation &obs, double dt)
    // 跟踪更新
    {
        const double delta = 2.0 * M_PI / static_cast<double>(plate_count_);

        // 判断哪个装甲板
        const double theta_pred = theta_ + omega_ * dt;
        int best_i = 0;
        double best_r = wrapAngle(obs.alpha - theta_pred);
        for (int i = 1; i < plate_count_; ++i)
        {
            const double r = wrapAngle(obs.alpha - theta_pred -
                                       i * delta);
            if (std::fabs(r) < std::fabs(best_r))
            {
                best_i = i;
                best_r = r;
            }
        }

        // 滞回
        const double r_cur = wrapAngle(obs.alpha - theta_pred - face_ * delta);
        if (std::fabs(r_cur) <= std::fabs(best_r) + face_hysteresis_)
        {
            best_i = face_;
            best_r = r_cur;
        }

        //更新数据        
        theta_ = theta_pred + alpha_theta_ * best_r;
        if (dt > 1e-4)
            omega_ += beta_theta_ * (best_r / dt);
        if (omega_ > max_omega_)
            omega_ = max_omega_;
        if (omega_ < -max_omega_)
            omega_ = -max_omega_;
        face_ = best_i;

        // 法向变化应该很小
        n_y_ = (1.0 - n_y_alpha_) * n_y_ + n_y_alpha_ * obs.n.y;

        // 2) 车体中心滤波
        const cv::Point3d C_pred = C_ + V_ * dt;
        const cv::Point3d e = obs.c - C_pred;
        C_ = C_pred + alpha_center_ * e;
        if (dt > 1e-4)
            V_ += beta_center_ * (e / dt);
        clampNorm(V_, max_speed_);
    }
    cv::Point3d SelectorNodelet::armorPosition(const cv::Point3d &C, double theta, int face) const 
    // 回推装甲板
    {
        const double delta = 2.0 * M_PI / static_cast<double>(plate_count_);
        const double phi = theta + face * delta;
        const double h = std::sqrt(std::max(0.0, 1.0 - n_y_ * n_y_));
        return C + cv::Point3d(radius_ * h * std::cos(phi),
                               -radius_ * n_y_,
                               radius_ * h * std::sin(phi));
    }
    cv::Point3d SelectorNodelet::predictArmor(double T, const cv::Point3d &p_current,
                                              const cv::Point3d &V, double theta,
                                              int face) const 
    // 状态预测

    {
        const double delta = 2.0 * M_PI / static_cast<double>(plate_count_);
        const double phi = theta + face * delta;
        const double phi_f = phi + omega_ * T;
        return p_current + V * T +
               cv::Point3d(radius_ * (std::cos(phi_f) - std::cos(phi)),
                           0.0,
                           radius_ * (std::sin(phi_f) - std::sin(phi)));
    }

    void SelectorNodelet::onDetection(const armor_detector::DetectionFrame::ConstPtr &msg)
    // 主执行
    {
        //收/算数据
        const std_msgs::Header &header = msg->header;
        const double t = header.stamp.toSec();
        if (t <= 0.0){return;}
        std::vector<Observation> obs;
        obs.reserve(msg->detections.size());
        for (const auto &d : msg->detections)  
        {
            Observation o;
            if (makeObservation(d, o))
                obs.push_back(o);
        }

        //判断状态
        if (!has_target_)
        {
            if (obs.empty()) {return;}
            //选/预/送
            auto best = std::max_element(
                obs.begin(), obs.end(),
                [](const Observation &a, const Observation &b)
                {
                    if (std::fabs(a.face_on - b.face_on) > 1e-6)
                        return a.face_on < b.face_on;
                    return a.p.z > b.p.z;
                });
            initializeTarget(*best, header, next_id_++);
            const cv::Point3d p_pred =predictArmor(predict_T_, best->p, V_, theta_, face_);
            const cv::Point3d p_base = best->p + v_obs_ * predict_T_;
            publishTarget(header, true, false, best->p, p_pred, p_base, C_, theta_, omega_, face_);
            return;
        }

        //看看时间
        double dt = t - state_stamp_;
        if (dt < 0.0)
        {
            ROS_WARN_THROTTLE(1.0, "selector: out-of-order stamp, ignored");
            return;
        }
        if (dt > lost_timeout_)
        {
            has_target_ = false;
            target_id_ = 0;
            publishInvalid(header);
            return;
        }

        // 认亲
        const cv::Point3d C_pred = C_ + V_ * dt;
        int best = -1;
        double best_err = gate_center_;
        for (int k = 0; k < static_cast<int>(obs.size()); ++k)
        {
            if (obs[k].red != target_red_){continue;}
            if (obs[k].face_on < face_on_min_){continue;}
            const cv::Point3d d = obs[k].c - C_pred;
            const double err = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
            if (err < best_err)
            {
                best_err = err;
                best = k;
            }
        }

        // 有端联想(有数据)
        if (best >= 0)
        {
            if (dt > dt_max_)
            {
                initializeTarget(obs[best], header, target_id_);
            }
            else
            {
                if (has_last_p_ && t - last_p_stamp_ > 1e-3 && t - last_p_stamp_ < 0.3)
                {
                    const cv::Point3d v_raw = (obs[best].p - last_p_obs_) / (t - last_p_stamp_);
                    v_obs_ = 0.5 * v_obs_ + 0.5 * v_raw;
                    clampNorm(v_obs_, 3.0);
                }
                last_p_obs_ = obs[best].p;
                last_p_stamp_ = t;
                has_last_p_ = true;

                updateTarget(obs[best], dt);
                state_stamp_ = t;
            }
            last_hit_stamp_ = t;
            
            cv::Point3d p_anchor = armorPosition(C_, theta_, face_);
            const cv::Point3d anchor_err = p_anchor - obs[best].p;
            if (std::sqrt(anchor_err.x*anchor_err.x+anchor_err.y*anchor_err.y+anchor_err.z * anchor_err.z) > 0.15)
                p_anchor = obs[best].p;
            const cv::Point3d p_pred = predictArmor(predict_T_, p_anchor, V_, theta_, face_);
            const cv::Point3d p_base = obs[best].p + v_obs_ * predict_T_;
            publishTarget(header, true, false, obs[best].p, p_pred, p_base, C_, theta_, omega_, face_);
            return;
        }

        // 无端联想(纯预测)
        if (t - last_hit_stamp_ > lost_timeout_)
        {
            has_target_ = false;
            target_id_ = 0;
            publishInvalid(header);
            return;
        }

        double coast = t - state_stamp_;
        if (coast < 0.0) {coast = 0.0;}

        const cv::Point3d Cc = C_ + V_ * coast;
        const double thc = theta_ + omega_ * coast;
        const cv::Point3d p_cur = armorPosition(Cc, thc, face_);
        const cv::Point3d p_pred = predictArmor(predict_T_, p_cur, V_, thc, face_);
        const cv::Point3d p_base = p_cur + v_obs_ * predict_T_;
        publishTarget(header, false, true, p_cur, p_pred, p_base, Cc, thc, omega_, face_);
    }

    void SelectorNodelet::publishTarget(
        const std_msgs::Header &header, bool valid, bool predicted,
        const cv::Point3d &position, const cv::Point3d &position_predict,
        const cv::Point3d &position_baseline, const cv::Point3d &car_center,
        double theta, double omega, int face)
    // 数据打包
    {
        armor_detector::TargetSelect out;
        out.header = header;
        out.valid = valid;
        out.predicted = predicted;
        out.target_id = target_id_;

        out.position = toPoint(position);
        out.position_predict = toPoint(position_predict);
        out.position_baseline_predict = toPoint(position_baseline);
        out.car_center = toPoint(car_center);

        out.phase = static_cast<float>(wrapAngle(theta));
        out.omega = static_cast<float>(omega);
        out.radius = static_cast<float>(radius_);
        out.face_index = static_cast<uint8_t>(face);
        out.plate_count = static_cast<uint8_t>(plate_count_);

        target_pub_.publish(out);
        publishMarkers(header, car_center, position, position_predict,
                       position_baseline, valid, predicted);
    }

    void SelectorNodelet::publishInvalid(const std_msgs::Header &header)
    // 跟丢清除
    {
        armor_detector::TargetSelect out;
        out.header = header;
        out.valid = false;
        out.predicted = false;
        out.target_id = 0;
        out.radius = static_cast<float>(radius_);
        out.plate_count = static_cast<uint8_t>(plate_count_);
        target_pub_.publish(out);

        visualization_msgs::MarkerArray arr;
        visualization_msgs::Marker m;
        m.header = header;
        m.ns = "target";
        m.action = visualization_msgs::Marker::DELETEALL;
        arr.markers.push_back(m);
        marker_pub_.publish(arr);
    }

    void SelectorNodelet::publishMarkers(
        const std_msgs::Header &header, const cv::Point3d &car_center,
        const cv::Point3d &current, const cv::Point3d &predicted,
        const cv::Point3d &baseline, bool valid, bool extrapolated)
    // 画在rviz里
    {
        std_msgs::Header h = header;
        if (h.frame_id.empty())
            h.frame_id = "camera_optical_frame";

        visualization_msgs::MarkerArray arr;
        auto add = [&](int id, int type, const cv::Point3d &p,
                       float r, float g, float b, float sx, float sy, float sz,
                       float a)
        {
            visualization_msgs::Marker m;
            m.header = h;
            m.ns = "target";
            m.id = id;
            m.type = type;
            m.action = visualization_msgs::Marker::ADD;
            m.pose.position = toPoint(p);
            m.pose.orientation.w = 1.0;
            m.scale.x = sx;
            m.scale.y = sy;
            m.scale.z = sz;
            m.color.r = r;
            m.color.g = g;
            m.color.b = b;
            m.color.a = a;
            m.lifetime = ros::Duration(1.0);
            arr.markers.push_back(m);
        };

        // 0：车体中心（蓝色球）
        add(0, visualization_msgs::Marker::SPHERE, car_center,
            0.1f, 0.4f, 1.0f, 0.14f, 0.14f, 0.14f, 0.9f);
        // 1：当前装甲板（绿色薄方块）
        add(1, visualization_msgs::Marker::CUBE, current,
            0.1f, 1.0f, 0.1f, 0.10f, 0.04f, 0.10f,
            valid ? 0.9f : 0.4f);
        // 2：预测装甲板（红色球）
        add(2, visualization_msgs::Marker::SPHERE, predicted,
            1.0f, 0.1f, 0.1f, 0.16f, 0.16f, 0.16f,
            extrapolated ? 0.9f : 0.6f);
        // 3：基础 position+velocity*T
        add(3, visualization_msgs::Marker::SPHERE, baseline,
            1.0f, 1.0f, 0.1f, 0.10f, 0.10f, 0.10f, 0.7f);

        marker_pub_.publish(arr);
    }
}

PLUGINLIB_EXPORT_CLASS(armor_detector::SelectorNodelet, nodelet::Nodelet)
