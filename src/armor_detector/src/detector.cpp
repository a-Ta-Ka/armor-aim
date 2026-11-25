#include "armor_detector/detector.h"
#include "armor_detector/geometry.h"

#include <pluginlib/class_list_macros.h>
#include <chrono>
#include <clocale>
#include <cmath>
#include <set>
#include <stdexcept>

namespace armor_detector
{
    void DetectorNodelet::onInit()
    {
        setlocale(LC_ALL, "");
        loadParams();
        setupInterfaces();
    }
    void DetectorNodelet::loadParams()
    {
        ros::NodeHandle &n = getPrivateNodeHandle();

        auto p_f = [&n](const char *key, float &v)
        {
            double d = v;
            n.param<double>(key, d, d);
            v = static_cast<float>(d);
        };
        
        std::string mode;
        n.param<std::string>("detection_mode", mode, "tv");
        if (mode == "tv" || mode == "nn")
        {
            detection_mode_ = mode;
        }
        else
        {
            ROS_WARN("Unknown detection_mode '%s', falling back to 'tv'. Valid values: 'tv', 'nn'.",
                     mode.c_str());
            detection_mode_ = "tv";
        }

        // 2. 神经网络参数
        n.param<std::string>("nn_model_path", nn_model_path_, "");
        p_f("nn_color_threshold", nn_color_threshold_);
        p_f("nn_number_threshold", nn_number_threshold_);
        p_f("nn_nms_threshold", nn_nms_threshold_);
        for (const float threshold : {nn_color_threshold_, nn_number_threshold_, nn_nms_threshold_})
        {
            if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f)
                throw std::invalid_argument("NN thresholds must be finite and in [0, 1].");
        }

        // 3. 传统视觉参数
        HsvRange &hsv = tv_params_.hsv;
        n.param<int>("BH_MIN", hsv.bh_min, hsv.bh_min);
        n.param<int>("BH_MAX", hsv.bh_max, hsv.bh_max);
        n.param<int>("RH_MIN", hsv.rh_min, hsv.rh_min);
        n.param<int>("RH_MAX", hsv.rh_max, hsv.rh_max);
        n.param<int>("RH2_MIN", hsv.rh2_min, hsv.rh2_min);
        n.param<int>("RH2_MAX", hsv.rh2_max, hsv.rh2_max);
        n.param<int>("S_MIN", hsv.s_min, hsv.s_min);
        n.param<int>("S_MAX", hsv.s_max, hsv.s_max);
        n.param<int>("V_MIN", hsv.v_min, hsv.v_min);
        n.param<int>("V_MAX", hsv.v_max, hsv.v_max);

        BarFilter &bar = tv_params_.bar;
        n.param<int>("min_area", bar.min_area, bar.min_area);
        n.param<int>("min_height", bar.min_height, bar.min_height);
        p_f("min_hw", bar.min_hw);
        p_f("max_hw", bar.max_hw);
        p_f("max_area", bar.max_area);
        p_f("min_fill", bar.min_fill);

        PairParam &pair = tv_params_.pair;
        p_f("dy_ratio", pair.dy_ratio);
        p_f("h_diff_max", pair.h_diff_max);
        p_f("w_ratio_max", pair.w_ratio_max);
        p_f("ang_diff_max", pair.ang_diff_max);
        p_f("wl_min", pair.wl_min);
        p_f("wl_max", pair.wl_max);
        p_f("endpoint_dy_ratio", pair.endpoint_dy_ratio);

        MirrorParam &mirror = tv_params_.mirror;
        n.param<bool>("mirror_enabled", mirror.enabled, mirror.enabled);
        p_f("mirror_dx_ratio", mirror.dx_ratio);
        p_f("mirror_w_ratio", mirror.w_ratio);
        p_f("mirror_x_tol_ratio", mirror.x_tol_ratio);
        p_f("mirror_min_gap_ratio", mirror.min_gap_ratio);
        p_f("mirror_max_gap_ratio", mirror.max_gap_ratio);
        p_f("mirror_max_height_ratio", mirror.max_height_ratio);

        // 4. 装甲板尺寸与大小判据
        n.param<double>("wl_large", wl_large_, wl_large_);
        n.param<double>("wl_small", wl_small_, wl_small_);
        n.param<double>("armor_large_w", armor_large_w_, armor_large_w_);
        n.param<double>("armor_large_l", armor_large_l_, armor_large_l_);
        n.param<double>("armor_small_w", armor_small_w_, armor_small_w_);
        n.param<double>("armor_small_l", armor_small_l_, armor_small_l_);

        pair.large_ratio = static_cast<float>(std::sqrt(wl_large_ * wl_small_));

        tv_detector_.setParams(tv_params_);
        nn_detector_.setLargeRatio(pair.large_ratio);

        // 5. 动态参数
        reconf_server_.reset(new dynamic_reconfigure::Server<Config>(n));
        reconf_server_->setCallback(boost::bind(&DetectorNodelet::onReconfigure, this, _1, _2));

        // 6. 神经网络模式
        if (detection_mode_ == "nn")
        {
            if (nn_model_path_.empty())
            {
                ROS_ERROR("detection_mode is 'nn', but 'nn_model_path' is empty!");
                throw std::runtime_error("NN model path is empty.");
            }
            try
            {
                ROS_INFO("Loading neural network model from: %s", nn_model_path_.c_str());
                nn_detector_.init(nn_model_path_, nn_color_threshold_,
                                  nn_number_threshold_, nn_nms_threshold_);
                ROS_INFO("Neural network model loaded successfully.");
            }
            catch (const std::exception &e)
            {
                NODELET_ERROR("Failed to load neural network model: %s", e.what());
                throw;
            }
        }

        ROS_INFO("Detector initialized in %s mode.", detection_mode_ == "nn" ? "NN" : "TV");
    }//读取参数
    void DetectorNodelet::setupInterfaces()
    {
        ros::NodeHandle &n = getPrivateNodeHandle();

        std::string input_topic = "/hk_camera/image_raw";
        std::string debug_topic = "/ft/debug/image";
        std::string detection_topic = "/ft/detection";
        std::string camera_info_topic = "/hk_camera/camera_info";
        int queue_size = 2;

        n.param<std::string>("input_topic", input_topic, input_topic);
        n.param<std::string>("debug_topic", debug_topic, debug_topic);
        n.param<std::string>("camera_info_topic", camera_info_topic, camera_info_topic);
        n.param<std::string>("detection_topic", detection_topic, detection_topic);
        n.param<int>("queue_size", queue_size, queue_size);

        image_transport::ImageTransport it(n);
        image_transport::TransportHints hints("compressed", ros::TransportHints(), n);

        debug_pub_ = n.advertise<sensor_msgs::Image>(debug_topic, queue_size);
        detection_pub_ = n.advertise<armor_detector::DetectionFrame>(detection_topic, queue_size);
        image_sub_ = it.subscribe(input_topic, queue_size, &DetectorNodelet::onImage, this, hints);
        info_sub_ = n.subscribe(camera_info_topic, queue_size, &DetectorNodelet::onCameraInfo, this);

        ROS_INFO("detector ready: in=%s out=%s camera_info=%s detection=%s",
                 input_topic.c_str(), debug_topic.c_str(),
                 camera_info_topic.c_str(), detection_topic.c_str());
    }//纯粹通信


    void DetectorNodelet::onCameraInfo(const sensor_msgs::CameraInfoConstPtr &msg)
    {
        D_.release();
        if (!msg->D.empty())
        {
            D_ = cv::Mat(msg->D.size(), 1, CV_64F, (void *)msg->D.data()).clone();
        }
        K_ = cv::Mat(3, 3, CV_64F, (void *)msg->K.data()).clone();
        camera_frame_ = msg->header.frame_id;
    }
    void DetectorNodelet::onImage(const sensor_msgs::ImageConstPtr &msg)
    {
        cur_header_ = msg->header;
        if (!camera_frame_.empty())
        {
            cur_header_.frame_id = camera_frame_;
        }
        dets_.clear();
        det_id_ = 0;

        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (const cv_bridge::Exception &e)
        {
            NODELET_ERROR_THROTTLE(1.0, "Image conversion failed: %s", e.what());
            armor_detector::DetectionFrame frame;
            frame.header = cur_header_;
            detection_pub_.publish(frame);
            return;
        }

        cv::Mat canvas = cv_ptr->image.clone();
        try
        {
            if (detection_mode_ == "nn")
                findWithNn(cv_ptr->image, canvas);
            else
                findWithTv(cv_ptr->image, canvas);
        }
        catch (const std::exception &e)
        {
            NODELET_ERROR_THROTTLE(1.0, "Detection failed in %s mode: %s",
                                   detection_mode_.c_str(), e.what());
            dets_.clear();
            canvas = cv_ptr->image.clone();
        }

        sensor_msgs::ImagePtr out = cv_bridge::CvImage(cur_header_, "bgr8", canvas).toImageMsg();
        debug_pub_.publish(out);

        armor_detector::DetectionFrame frame;
        frame.header = cur_header_;
        frame.detections = dets_;
        detection_pub_.publish(frame);
    }


    void DetectorNodelet::findWithTv(const cv::Mat &image, cv::Mat &canvas)
    {
        std::vector<Armor> cands;
        std::set<int> used;
        tv_detector_.find(image, canvas, cands);

        for (const Armor &armor : cands)
        {
            if (armor.mirrored)
            {
                used.insert(armor.bar_left);
                used.insert(armor.bar_right);
            }
        }

        for (const Armor &armor : cands)
        {
            if (armor.mirrored)
                continue;
            if (used.count(armor.bar_left) || used.count(armor.bar_right))
                continue;
            if (appendArmor(armor, canvas))
            {
                used.insert(armor.bar_left);
                used.insert(armor.bar_right);
            }
        }
    }
    void DetectorNodelet::findWithNn(const cv::Mat &image, cv::Mat &canvas)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<Armor> armors = nn_detector_.find(image);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();

        ++nn_frames_;
        if (nn_frames_ > 10) // 前 10 帧含首次推理的初始化开销，不计入平均
        {
            nn_total_ms_ += ms;
            ++nn_timed_frames_;
            if (nn_timed_frames_ % 100 == 0)
            {
                NODELET_INFO("NN detect mean: %.2f ms over %zu frames (10 warmup frames excluded).",
                             nn_total_ms_ / nn_timed_frames_, nn_timed_frames_);
            }
        }

        for (const Armor &armor : armors)
            appendArmor(armor, canvas);
    }


    bool DetectorNodelet::appendArmor(const Armor &armor, cv::Mat &canvas)
    {
        const cv::Scalar center_color = armor.is_red ? cv::Scalar(0, 0, 255)
                                                     : cv::Scalar(0, 255, 0);
        const int armor_id = det_id_++;
        std::string tag = std::string(armor.is_large ? "L" : "S") + std::to_string(armor_id);

        if (armor.number >= 0 && armor.number < 9)
        {
            static const char *number_names[9] = {"guard", "1", "2", "3", "4", "5", "outpost", "base", "base"};
            tag += " " + std::string(armor.is_red ? "red:" : "blue:") + number_names[armor.number];
        }
        //位姿解算
        cv::Mat rvec, tvec;
        bool pose_ok = false;
        if (!K_.empty())
            pose_ok = estimatePose(armor, rvec, tvec);
        if (pose_ok)
        {
            const double rot_deg = std::sqrt(rvec.at<double>(0) * rvec.at<double>(0) +
                                             rvec.at<double>(1) * rvec.at<double>(1) +
                                             rvec.at<double>(2) * rvec.at<double>(2)) *57.3;
            ROS_DEBUG_THROTTLE(0.5, "[%s] t=(%.2f,%.2f,%.2f) |r|=%.1fdeg",
                               tag.c_str(), tvec.at<double>(0), tvec.at<double>(1),
                               tvec.at<double>(2), rot_deg);

            //发 TF 和填 Detection
            const tf2::Quaternion q = toQuaternion(rvec);
            broadcastTf(armor_id, tvec, q);
            makeDetection(armor, armor_id, tvec, q);
        }

        //画画
        drawArmor(canvas, armor.q, armor.center, center_color,
                  pose_ok, rvec, tvec, K_, D_);
        cv::putText(canvas, tag,
                    cv::Point((int)armor.q[0].x - 10, (int)armor.q[0].y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, center_color, 2);
        return pose_ok;
    }//位姿解算和发送和画画
    bool DetectorNodelet::estimatePose(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec) const
    {
        const double armor_w = armor.is_large ? armor_large_w_ : armor_small_w_;
        const double armor_l = armor.is_large ? armor_large_l_ : armor_small_l_;
        return armor_detector::estimatePose(armor.q, armor_w, armor_l, K_, D_, rvec, tvec);
    }//pnp解算
    void DetectorNodelet::broadcastTf(int id, const cv::Mat &tvec, const tf2::Quaternion &q)
    {
        geometry_msgs::TransformStamped tf_msg;
        tf_msg.header = cur_header_; // 图像时间戳
        tf_msg.child_frame_id = "armor_" + std::to_string(id);
        tf_msg.transform.translation.x = tvec.at<double>(0);
        tf_msg.transform.translation.y = tvec.at<double>(1);
        tf_msg.transform.translation.z = tvec.at<double>(2);
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();
        tf_broadcaster_.sendTransform(tf_msg);
    }//发给rviz
    void DetectorNodelet::makeDetection(const Armor &armor, int id,
                                        const cv::Mat &tvec, const tf2::Quaternion &q)
    {
        armor_detector::Detection d;
        d.id = id;
        d.is_red = armor.is_red;
        d.is_large_armor = armor.is_large;
        d.pose.position.x = tvec.at<double>(0);
        d.pose.position.y = tvec.at<double>(1);
        d.pose.position.z = tvec.at<double>(2);
        d.pose.orientation.x = q.x();
        d.pose.orientation.y = q.y();
        d.pose.orientation.z = q.z();
        d.pose.orientation.w = q.w();
        for (int n = 0; n < 4; ++n)
        {
            d.corners[n].x = armor.q[n].x;
            d.corners[n].y = armor.q[n].y;
            d.corners[n].z = 0.0f;
        }
        d.confidence = armor.confidence;
        dets_.push_back(d);
    }//发给跟踪器
    void DetectorNodelet::onReconfigure(Config &config, uint32_t level)
    {
        (void)level;

        tv_params_.hsv.bh_min = config.BH_MIN;
        tv_params_.hsv.bh_max = config.BH_MAX;
        tv_params_.hsv.rh_min = config.RH_MIN;
        tv_params_.hsv.rh_max = config.RH_MAX;
        tv_params_.hsv.rh2_min = config.RH2_MIN;
        tv_params_.hsv.rh2_max = config.RH2_MAX;
        tv_params_.hsv.s_min = config.S_MIN;
        tv_params_.hsv.s_max = config.S_MAX;
        tv_params_.hsv.v_min = config.V_MIN;
        tv_params_.hsv.v_max = config.V_MAX;

        tv_params_.bar.min_area = config.min_area;
        tv_params_.bar.min_height = config.min_height;
        tv_params_.bar.min_fill = static_cast<float>(config.min_fill);

        tv_params_.pair.wl_max = static_cast<float>(config.wl_max);
        tv_params_.pair.ang_diff_max = static_cast<float>(config.ang_diff_max);
        tv_params_.pair.h_diff_max = static_cast<float>(config.h_diff_max);
        tv_params_.pair.w_ratio_max = static_cast<float>(config.w_ratio_max);

        tv_detector_.setParams(tv_params_);
    }//动态参数

} // namespace armor_detector

PLUGINLIB_EXPORT_CLASS(armor_detector::DetectorNodelet, nodelet::Nodelet)
