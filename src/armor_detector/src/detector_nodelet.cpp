#include "armor_detector/detector_nodelet.h"
#include <pluginlib/class_list_macros.h>
#include <algorithm>


namespace armor_detector
{
    namespace // 匿名 namespace，这样做只是单纯想隐藏这些辅助函数，好看一点
    {
        struct Pair // 灯条配对组
        {
            int i, j;
            float cost;
            bool large;
        };
        static cv::RotatedRect normalizeRect(const cv::RotatedRect &rr_in)//规范旋转矩形

        {
            cv::RotatedRect rr = rr_in;
            float ang = rr.angle;
            if (rr.size.width > rr.size.height)
            {
                std::swap(rr.size.width, rr.size.height);
                ang += 90.0f;
            }
            while (ang >= 90.0f)
                ang -= 180.0f;
            while (ang < -90.0f)
                ang += 180.0f;
            rr.angle = ang;
            return rr;
        }
        static bool pointcompare(const cv::Point2f &a, const cv::Point2f &b)//灯条上下点
        {
            return a.y < b.y;
        }
        static bool paircompare(const Pair &a, const Pair &b) // 贪心比较
        {
            return a.cost < b.cost;
        }
        static void sortBarPoints(const cv::Point2f in[4], cv::Point2f &top, cv::Point2f &bottom)//排点
        {
            std::vector<cv::Point2f> v(in, in + 4);
            std::sort(v.begin(), v.end(), pointcompare);

            top = (v[0] + v[1]) * 0.5f;
            bottom = (v[2] + v[3]) * 0.5f;
        }
        static void findBars(const cv::Mat &mask, const BarFilter &p, std::vector<cv::RotatedRect> &bars)//找个灯条
        {
            bars.clear();
            std::vector<std::vector<cv::Point>> contours; // 详细点
            std::vector<cv::Vec4i> hierarchy;//关系
            cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            bars.reserve(contours.size());

            for (size_t i = 0; i < contours.size(); i++)
            {
                if (cv::contourArea(contours[i]) <= p.min_area)  //去噪！
                    continue;
                cv::RotatedRect rr = normalizeRect(cv::minAreaRect(contours[i]));
                const float w = rr.size.width, h = rr.size.height; // 归一化后 w <= h
                if (h < p.min_height)
                    continue; // 矮
                if (h / w < p.min_hw || h / w > p.max_hw)
                    continue; // 不细长
                if (w * h > p.max_area)
                    continue; // 超级大灯
                bars.push_back(rr);
            }
        }
        static void pairBars(const std::vector<cv::RotatedRect> &bars,
                             const PairParam &p, std::vector<Pair> &cands)//配对灯条
        {
            cands.clear();
            for (int i = 0; i < (int)bars.size(); i++)
            {
                for (int j = i + 1; j < (int)bars.size(); j++)
                {
                    const cv::RotatedRect &a = bars[i];
                    const cv::RotatedRect &b = bars[j];

                    const float max_len = std::max(a.size.height, b.size.height);
                    const float avg_len = (a.size.height + b.size.height) * 0.5f;
                    if (max_len <= 0.0f || avg_len <= 0.0f)
                        continue;

                    // 1. 上下高度差
                    const float dy = std::abs(a.center.y - b.center.y);
                    if (dy > p.dy_ratio * max_len)
                        continue;

                    // 2. 长度一致
                    const float len_diff = std::abs(a.size.height - b.size.height) / max_len;
                    if (len_diff > p.len_diff_max)
                        continue;

                    // 3. 宽度一致
                    const float avg_w = (a.size.width + b.size.width) * 0.5f;
                    if (avg_w <= 0.0f)
                        continue;
                    const float w_diff = std::abs(a.size.width - b.size.width) / avg_w;
                    if (w_diff > p.w_diff_max)
                        continue;

                    // 4. 平行
                    float d_ang = std::abs(a.angle - b.angle);
                    if (d_ang > 90.0f)
                        d_ang = 180.0f - d_ang;
                    if (d_ang > p.ang_diff_max)
                        continue;

                    // 5. 中心距
                    const float dx = std::abs(a.center.x - b.center.x);
                    const float ratio = dx / avg_len;
                    if (ratio < p.wl_min || ratio > p.wl_max)
                        continue;

                    const float cost = d_ang * 2.0f + len_diff * 40.0f + w_diff * 20.0f + dy / max_len * 20.0f;
                    cands.push_back({i, j, cost, ratio >= p.large_ratio});
                }
            }
            std::sort(cands.begin(), cands.end(), paircompare);
        }
        static bool solveArmorPose(const cv::Point2f quad[4],
                                   double armor_w, double armor_l,
                                   const cv::Mat &K, const cv::Mat &D,
                                   cv::Mat &rvec, cv::Mat &tvec)//位姿的解算
        {
            const double half_w = armor_w * 0.5;
            const double half_h = armor_l * 0.5;
            const std::vector<cv::Point3f> obj = {
                {(float)-half_w, (float)-half_h, 0.0f}, // 左上
                {(float)half_w, (float)-half_h, 0.0f},  // 右上
                {(float)half_w, (float)half_h, 0.0f},   // 右下
                {(float)-half_w, (float)half_h, 0.0f},  // 左下
            };
            const std::vector<cv::Point2f> img(quad, quad + 4);

            return cv::solvePnP(obj, img, K, D, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        }
        static void drawArmor(cv::Mat &img,
                              const cv::Point2f quad[4],
                              const cv::Point2f &center,
                              const cv::Scalar &center_color,
                              bool pose_ok,
                              const cv::Mat &rvec, const cv::Mat &tvec,
                              const cv::Mat &K, const cv::Mat &D)//绘画专区
        {
            // 1. 装甲板四角点（紫框 + 青点）
            for (int m = 0; m < 4; m++)
            {
                cv::line(img, quad[m], quad[(m + 1) % 4], cv::Scalar(255, 0, 255), 2);
            }
            // 2. 坐标轴和中心点
            if (pose_ok)
            {
                std::vector<cv::Point3f> axis = {{0, 0, 0}, {0.05f, 0, 0}, {0, 0.05f, 0}, {0, 0, 0.05f}};
                std::vector<cv::Point2f> proj;
                cv::projectPoints(axis, rvec, tvec, K, D, proj);
                cv::circle(img, center, 5, center_color, -1);
                cv::line(img, proj[0], proj[1], cv::Scalar(0, 0, 255), 2);
                cv::line(img, proj[0], proj[2], cv::Scalar(0, 255, 0), 2);
                cv::line(img, proj[0], proj[3], cv::Scalar(255, 0, 0), 2);
            }
        }
    } // anonymous namespace

    void DetectorNodelet::onInit()
    {
        setlocale(LC_ALL, "");
        ros::NodeHandle &n = getPrivateNodeHandle();
        auto p_f = [&n](const char *key, float &v)    //让param能读float
        {
            double d = v;
            n.param<double>(key, d, d);
            v = static_cast<float>(d);
        };

        std::string input_topic = "/hk_camera/image_raw";
        std::string debug_topic = "/ft/debug/image";
        std::string detection_topic = "/ft/detection";
        std::string camera_info_topic = "/hk_camera/camera_info";
        int queue_size = 1;

        n.param<std::string>("input_topic", input_topic, input_topic);
        n.param<std::string>("debug_topic", debug_topic, debug_topic);
        n.param<std::string>("camera_info_topic", camera_info_topic,camera_info_topic);
        n.param<std::string>("detection_topic", detection_topic, detection_topic);
        n.param<int>("queue_size", queue_size, queue_size);

        n.param<int>("BH_MIN", BH_MIN_, BH_MIN_);
        n.param<int>("BH_MAX", BH_MAX_, BH_MAX_);
        n.param<int>("RH_MIN", RH_MIN_, RH_MIN_);
        n.param<int>("RH_MAX", RH_MAX_, RH_MAX_);
        n.param<int>("RH2_MIN", RH2_MIN_, RH2_MIN_);
        n.param<int>("RH2_MAX", RH2_MAX_, RH2_MAX_);
        n.param<int>("S_MIN", S_MIN_, S_MIN_);
        n.param<int>("S_MAX", S_MAX_, S_MAX_);
        n.param<int>("V_MIN", V_MIN_, V_MIN_);
        n.param<int>("V_MAX", V_MAX_, V_MAX_);

        n.param<int>("min_area", bar_filter_.min_area, bar_filter_.min_area);
        n.param<int>("min_height", bar_filter_.min_height, bar_filter_.min_height);
        p_f("min_hw", bar_filter_.min_hw);
        p_f("max_hw", bar_filter_.max_hw);
        p_f("max_area", bar_filter_.max_area);

        p_f("dy_ratio", pair_param_.dy_ratio);
        p_f("len_diff_max", pair_param_.len_diff_max);
        p_f("w_diff_max", pair_param_.w_diff_max);
        p_f("ang_diff_max", pair_param_.ang_diff_max);
        p_f("wl_min", pair_param_.wl_min);
        p_f("wl_max", pair_param_.wl_max);

        n.param<double>("wl_large", wl_large, wl_large);
        n.param<double>("wl_small", wl_small, wl_small);
        n.param<double>("armor_large_w", armor_large_w_, armor_large_w_);
        n.param<double>("armor_large_l", armor_large_l_, armor_large_l_);
        n.param<double>("armor_small_w", armor_small_w_, armor_small_w_);
        n.param<double>("armor_small_l", armor_small_l_, armor_small_l_);

        pair_param_.large_ratio = static_cast<float>(std::sqrt(wl_large * wl_small));
        kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

        image_transport::ImageTransport it(n);
        image_transport::TransportHints hints("compressed", ros::TransportHints(), n);

        debug_pub_ = n.advertise<sensor_msgs::Image>(debug_topic, 1);
        detection_pub_ = n.advertise<armor_detector::DetectionFrame>(detection_topic, 1);
        image_sub_ = it.subscribe(input_topic, queue_size, &DetectorNodelet::imageCallback, this, hints);
        info_sub_ = n.subscribe(camera_info_topic, queue_size, &DetectorNodelet::cameraInfoCallback, this);

        ROS_INFO("detector ready: in=%s out=%s camera_info=%s detection=%s",
                 input_topic.c_str(), debug_topic.c_str(),
                 camera_info_topic.c_str(), detection_topic.c_str());
    }

    void DetectorNodelet::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg)
    {
        K_ = cv::Mat(3, 3, CV_64F, (void *)msg->K.data()).clone();

        D_.release();
        if (!msg->D.empty())
        {
            D_ = cv::Mat(msg->D.size(), 1, CV_64F, (void *)msg->D.data()).clone();
        }


    }
    void DetectorNodelet::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        cur_header_ = msg->header;
        dets_.clear();
        det_id_ = 0;
        
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);

        cv::Mat canvas = cv_ptr->image.clone();
        update(canvas);

        sensor_msgs::ImagePtr out =
            cv_bridge::CvImage(msg->header, "bgr8", canvas).toImageMsg();
        debug_pub_.publish(out);

        armor_detector::DetectionFrame frame;
        frame.header = msg->header;
        frame.detections = dets_;
        detection_pub_.publish(frame);
    }

    void DetectorNodelet::update(cv::Mat &g_img)
    {
        cv::Mat hsv, blur;
        cv::GaussianBlur(g_img, blur, cv::Size(3, 3), 0);
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
        cv::Mat red_high;
        cv::inRange(hsv,
                    cv::Scalar(RH2_MIN_, S_MIN_, V_MIN_),
                    cv::Scalar(RH2_MAX_, S_MAX_, V_MAX_),
                    red_high);

        processColor(hsv,
                     cv::Scalar(BH_MIN_, S_MIN_, V_MIN_),
                     cv::Scalar(BH_MAX_, S_MAX_, V_MAX_),
                     g_img, false);
        processColor(hsv,
                     cv::Scalar(RH_MIN_, S_MIN_, V_MIN_),
                     cv::Scalar(RH_MAX_, S_MAX_, V_MAX_),
                     g_img, true, red_high);
    }
    void DetectorNodelet::processColor(const cv::Mat &hsv,
                                       const cv::Scalar &lo, const cv::Scalar &hi,
                                       cv::Mat &g_img, bool is_red_armor,
                                       const cv::Mat &mask_extra)
    {
        cv::Mat mask;
        cv::inRange(hsv, lo, hi, mask);
        if (!mask_extra.empty())
            cv::bitwise_or(mask, mask_extra, mask);
        cv::dilate(mask, mask, kernel_);
        cv::erode(mask, mask, kernel_);
        detectArmors(mask, g_img, is_red_armor);
    }
    void DetectorNodelet::detectArmors(const cv::Mat &g_mask, cv::Mat &g_img, bool is_red_armor)
    {
        std::vector<cv::RotatedRect> bars;
        findBars(g_mask, bar_filter_, bars);

        std::vector<Pair> cands;
        pairBars(bars, pair_param_, cands);

        std::vector<bool> used(bars.size(), false);
        const cv::Scalar center_color = is_red_armor ? cv::Scalar(0, 0, 255)
                                                     : cv::Scalar(0, 255, 0);

        for (size_t k = 0; k < cands.size(); k++)
        {
            const Pair &c = cands[k];
            if (used[c.i] || used[c.j])
                continue;

            // 左右灯条（取上下端点）
            const cv::RotatedRect &bi = bars[c.i];
            const cv::RotatedRect &bj = bars[c.j];
            const bool i_is_left = bi.center.x < bj.center.x;
            const cv::RotatedRect &bL = i_is_left ? bi : bj;
            const cv::RotatedRect &bR = i_is_left ? bj : bi;

            cv::Point2f pL_top, pL_bot, pR_top, pR_bot;
            cv::Point2f ptmp[4];
            bL.points(ptmp);
            sortBarPoints(ptmp, pL_top, pL_bot);
            bR.points(ptmp);
            sortBarPoints(ptmp, pR_top, pR_bot);

            // 端点对齐：半截灯条 / 错配 在这里拦下（宁漏检）
            const float h_avg = (bL.size.height + bR.size.height) * 0.5f;
            if (std::abs(pL_top.y - pR_top.y) > 0.35f * h_avg)
                continue;
            if (std::abs(pL_bot.y - pR_bot.y) > 0.35f * h_avg)
                continue;

            used[c.i] = true; // ← 全部校验通过后才占用灯条
            used[c.j] = true;

            const int armor_id = det_id_++; 
            const std::string tag = std::string(c.large ? "L" : "S") + std::to_string(armor_id);

            // 位姿解算
            const cv::Point2f quad[4] = {pL_top, pR_top, pR_bot, pL_bot};
            cv::Mat rvec, tvec;
            bool pose_ok = false;
            if (!K_.empty())
            {
                const double armor_w = c.large ? armor_large_w_ : armor_small_w_;
                const double armor_l = c.large ? armor_large_l_ : armor_small_l_;
                pose_ok = solveArmorPose(quad, armor_w, armor_l, K_, D_, rvec, tvec);
            }

            if (pose_ok)
            {
                const double rot_deg = std::sqrt(rvec.at<double>(0) * rvec.at<double>(0) +
                                                 rvec.at<double>(1) * rvec.at<double>(1) +
                                                 rvec.at<double>(2) * rvec.at<double>(2)) *
                                       57.3;
                ROS_DEBUG_THROTTLE(0.5, "[%s] t=(%.2f,%.2f,%.2f) |r|=%.1fdeg",
                                   tag.c_str(), tvec.at<double>(0), tvec.at<double>(1),
                                   tvec.at<double>(2), rot_deg);

                cv::Mat R;
                cv::Rodrigues(rvec, R);
                tf2::Matrix3x3 m(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                                 R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                                 R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
                tf2::Quaternion q;
                m.getRotation(q);

                geometry_msgs::TransformStamped tf_msg;
                tf_msg.header = cur_header_; // 图像时间戳
                tf_msg.child_frame_id = "armor_" + std::to_string(armor_id);
                tf_msg.transform.translation.x = tvec.at<double>(0);
                tf_msg.transform.translation.y = tvec.at<double>(1);
                tf_msg.transform.translation.z = tvec.at<double>(2);
                tf_msg.transform.rotation.x = q.x();
                tf_msg.transform.rotation.y = q.y();
                tf_msg.transform.rotation.z = q.z();
                tf_msg.transform.rotation.w = q.w();
                tf_broadcaster_.sendTransform(tf_msg);

                armor_detector::Detection d;
                d.id = armor_id;
                d.is_red = is_red_armor;
                d.is_large_armor = c.large;
                d.pose.position.x = tvec.at<double>(0);
                d.pose.position.y = tvec.at<double>(1);
                d.pose.position.z = tvec.at<double>(2);
                d.pose.orientation.x = q.x();
                d.pose.orientation.y = q.y();
                d.pose.orientation.z = q.z();
                d.pose.orientation.w = q.w();
                for (int n = 0; n < 4; ++n)
                {
                    d.corners[n].x = quad[n].x;
                    d.corners[n].y = quad[n].y;
                    d.corners[n].z = 0.0f;
                }
                d.confidence = 1.0f;
                dets_.push_back(d);
            }

            // 绘制
            const cv::Point2f armor_center = (bi.center + bj.center) * 0.5f;
            drawArmor(g_img, quad, armor_center, center_color, pose_ok, rvec, tvec, K_, D_);
            cv::putText(g_img, tag,
                        cv::Point((int)quad[0].x - 10, (int)quad[0].y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, center_color, 2);
        }
    }
} // namespace armor_detector

PLUGINLIB_EXPORT_CLASS(armor_detector::DetectorNodelet, nodelet::Nodelet)
