#include "armor_detector/tracker.h"
#include <pluginlib/class_list_macros.h>
#include <algorithm>
#include <limits>

namespace armor_detector
{
    void TrackerNodelet::onInit()
    {
        ros::NodeHandle &n = getPrivateNodeHandle();
        n.param<int>("queue_size", queue_size_, queue_size_);
        n.param<std::string>("detection_topic", detection_topic_, detection_topic_);
        n.param<std::string>("track_topic", track_topic_, track_topic_);
        n.param<std::string>("marker_topic", marker_topic_, marker_topic_);

        n.param<double>("predict_T", predict_t_, predict_t_);
        n.param<double>("max_match_dist", max_match_dist_, max_match_dist_);
        n.param<double>("lost_timeout", lost_timeout_, lost_timeout_);
        n.param<double>("dt_max", dt_max_, dt_max_);
        n.param<double>("q_pos", q_pos_, q_pos_);
        n.param<double>("q_vel", q_vel_, q_vel_);
        n.param<double>("r_pos", r_pos_, r_pos_);
        n.param<double>("init_pos_var", init_pos_var_, init_pos_var_);
        n.param<double>("init_vel_var", init_vel_var_, init_vel_var_);

        det_sub_ = n.subscribe(detection_topic_, queue_size_, &TrackerNodelet::onDetection, this);
        track_pub_ = n.advertise<armor_detector::TrackTarget>(track_topic_, 1);
        marker_pub_ = n.advertise<visualization_msgs::Marker>(marker_topic_, 1);
    }

    void TrackerNodelet::onDetection(const armor_detector::DetectionFrame::ConstPtr &msg)
    {
        const double t = msg->header.stamp.toSec();
        const double dt = (last_stamp_ > 0.0) ? (t - last_stamp_) : -1.0;
        last_stamp_ = t;

        if (dt <= 1e-3 || dt > dt_max_)
        {
            if (msg->detections.empty())
            {
                has_track_ = false;
                last_hit_stamp_ = -1.0;
                return;
            }
            const auto &p = msg->detections.front().pose.position;
            resetFilter(cv::Point3d(p.x, p.y, p.z));
            has_track_ = true;
            ++track_id_;
            last_hit_stamp_ = t;

            ROS_INFO("tracker: new track id=%d (dt=%.3f)", track_id_, dt);
            publishTrack(msg->header, true);
            return;
        }//错时
        if (has_track_ && last_hit_stamp_ >= 0.0 && (t - last_hit_stamp_) > lost_timeout_)
        {
            ROS_INFO("tracker: track id=%d lost", track_id_);
            has_track_ = false;
            last_hit_stamp_ = -1.0;
        }//超时
        if (!has_track_)
        {
            if (msg->detections.empty()) return;
            const auto it = std::min_element(
                msg->detections.begin(), msg->detections.end(),
                [](const armor_detector::Detection &a, const armor_detector::Detection &b)
                { return a.pose.position.z < b.pose.position.z; });

            const auto &p = it->pose.position;
            resetFilter(cv::Point3d(p.x, p.y, p.z));
            has_track_ = true;
            ++track_id_;
            last_hit_stamp_ = t;
            ROS_INFO("tracker: new track id=%d", track_id_);
            publishTrack(msg->header, true);
            return;
        }//建轨

        //预测
        kf_.transitionMatrix.at<double>(0, 3) = dt;
        kf_.transitionMatrix.at<double>(1, 4) = dt;
        kf_.transitionMatrix.at<double>(2, 5) = dt;
        const cv::Mat pred = kf_.predict();

        //关联
        double best_err = 0.0;
        const armor_detector::Detection *best =
            associate(pred, msg->detections, best_err);

        //门限
        const bool hit = (best != nullptr);
        if (hit)
        {
            const cv::Mat meas = (cv::Mat_<double>(3, 1) << best->pose.position.x,
                                  best->pose.position.y,
                                  best->pose.position.z);
            kf_.correct(meas);
            last_hit_stamp_ = t;
        }
        else if (!msg->detections.empty()) ROS_WARN_THROTTLE(1.0, "tracker:nearest%.2f m,ignore",best_err);
        publishTrack(msg->header, hit);
    }

    const armor_detector::Detection *TrackerNodelet::associate(
        const cv::Mat &pred,
        const std::vector<armor_detector::Detection> &dets,
        double &best_err) const
    {
        const armor_detector::Detection *best = nullptr;
        best_err = std::numeric_limits<double>::max();

        for (const auto &d : dets)
        {
            const double err = cv::norm(
                cv::Point3d(d.pose.position.x, d.pose.position.y, d.pose.position.z) -
                cv::Point3d(pred.at<double>(0), pred.at<double>(1), pred.at<double>(2)));
            if (err < best_err)
            {
                best_err = err;
                best = &d;
            }
        }

        if (best && best_err < max_match_dist_)
            return best;
        return nullptr;
    }//贪心匹配
    void TrackerNodelet::resetFilter(const cv::Point3d &p)
    {
        kf_.init(6, 3, 0, CV_64F);

        cv::setIdentity(kf_.transitionMatrix);  // F，dt 在回调里逐帧填
        cv::setIdentity(kf_.measurementMatrix); // H = [I3 | 0]

        // x = (p, 0)：位置取首帧观测，速度未知先置 0
        kf_.statePost.at<double>(0) = p.x;
        kf_.statePost.at<double>(1) = p.y;
        kf_.statePost.at<double>(2) = p.z;
        kf_.statePost.at<double>(3) = 0.0;
        kf_.statePost.at<double>(4) = 0.0;
        kf_.statePost.at<double>(5) = 0.0;

        // 初始协方差
        kf_.errorCovPost = cv::Mat::zeros(6, 6, CV_64F);
        for (int i = 0; i < 3; ++i)
        {
            kf_.errorCovPost.at<double>(i, i) = init_pos_var_;
            kf_.errorCovPost.at<double>(i + 3, i + 3) = init_vel_var_;
        }

        // Q：位置过程噪声小、速度过程噪声大
        kf_.processNoiseCov = cv::Mat::zeros(6, 6, CV_64F);
        for (int i = 0; i < 3; ++i)
        {
            kf_.processNoiseCov.at<double>(i, i) = q_pos_;
            kf_.processNoiseCov.at<double>(i + 3, i + 3) = q_vel_;
        }

        // R：观测噪声，三个方向同方差
        kf_.measurementNoiseCov = cv::Mat::eye(3, 3, CV_64F) * r_pos_;
    }//建轨初始化

    void TrackerNodelet::publishTrack(const std_msgs::Header &header, bool tracking)
    {
        armor_detector::TrackTarget out;
        out.header = header;
        out.id = track_id_;
        out.tracking = tracking;

        const cv::Mat &s = kf_.statePost;
        out.position.x = s.at<double>(0);
        out.position.y = s.at<double>(1);
        out.position.z = s.at<double>(2);
        out.velocity.x = s.at<double>(3);
        out.velocity.y = s.at<double>(4);
        out.velocity.z = s.at<double>(5);

        out.position_predict.x = out.position.x + out.velocity.x * predict_t_;
        out.position_predict.y = out.position.y + out.velocity.y * predict_t_;
        out.position_predict.z = out.position.z + out.velocity.z * predict_t_;

        track_pub_.publish(out);
        publishMarker(header, tracking, out.position_predict);
    }//发送追踪坐标
    void TrackerNodelet::publishMarker(const std_msgs::Header &header, bool tracking,
                                       const geometry_msgs::Point &p)
    {
        visualization_msgs::Marker m;
        m.header = header;
        m.ns = "track";
        m.id = 0;
        m.type = visualization_msgs::Marker::SPHERE;
        m.action = visualization_msgs::Marker::ADD;

        m.pose.position = p;
        m.pose.orientation.w = 1.0; 
        m.scale.x = m.scale.y = m.scale.z = 0.1; 

        m.color.r = tracking ? 0.0f : 1.0f;
        m.color.g = tracking ? 1.0f : 0.8f;
        m.color.b = 0.0f;
        m.color.a = 0.8f;

        m.lifetime = ros::Duration(0.3);
        marker_pub_.publish(m);
    }//rviz画小球
}

PLUGINLIB_EXPORT_CLASS(armor_detector::TrackerNodelet, nodelet::Nodelet)
