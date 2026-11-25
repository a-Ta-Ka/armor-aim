#include "armor_detector/tv_detector.h"
#include "armor_detector/geometry.h"
#include <algorithm>
#include <cmath>

namespace armor_detector
{
    namespace
    {
        cv::RotatedRect normalizeRect(const cv::RotatedRect &rr_in)
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
        }//规范旋转矩形，长边h，角度[-90, 90)
        bool pointCompare(const cv::Point2f &a, const cv::Point2f &b){return a.y < b.y;}
        void sortBarPoints(const cv::Point2f in[4], cv::Point2f &top, cv::Point2f &bottom)
        {
            std::vector<cv::Point2f> v(in, in + 4);
            std::sort(v.begin(), v.end(), pointCompare);
            top = (v[0] + v[1]) * 0.5f;
            bottom = (v[2] + v[3]) * 0.5f;
        }//按 y 排序，输出上下中点
        template <typename PairT>
        bool byCost(const PairT &a, const PairT &b)
        {
            return a.cost < b.cost;
        } //贪心排序
    } // namespace

    TvDetector::TvDetector()
    {
        kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    }
    void TvDetector::maskForColor(const cv::Mat &hsv,
                                  const cv::Scalar &lo, const cv::Scalar &hi,
                                  const cv::Mat &mask_extra, cv::Mat &mask)
    {
        cv::inRange(hsv, lo, hi, mask);
        if (!mask_extra.empty())
            cv::bitwise_or(mask, mask_extra, mask);
        cv::dilate(mask, mask, kernel_);
        cv::erode(mask, mask, kernel_);
    }//图像处理
    void TvDetector::find(const cv::Mat &bgr, cv::Mat &canvas, std::vector<Armor> &out)
    {
        cv::Mat blur, hsv, red_extra, mask;
        cv::GaussianBlur(bgr, blur, cv::Size(3, 3), 0);
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

        const HsvRange &h = params_.hsv;
        cv::inRange(hsv,
                    cv::Scalar(h.rh2_min, h.s_min, h.v_min),
                    cv::Scalar(h.rh2_max, h.s_max, h.v_max),
                    red_extra);
        maskForColor(hsv,
                     cv::Scalar(h.bh_min, h.s_min, h.v_min),
                     cv::Scalar(h.bh_max, h.s_max, h.v_max),
                     cv::Mat(), mask);
        detectInMask(mask, false, canvas, out);
        maskForColor(hsv,
                     cv::Scalar(h.rh_min, h.s_min, h.v_min),
                     cv::Scalar(h.rh_max, h.s_max, h.v_max),
                     red_extra, mask);
        detectInMask(mask, true, canvas, out);
    }//颜色识别
    void TvDetector::detectInMask(const cv::Mat &mask, bool is_red,
                                  cv::Mat &canvas, std::vector<Armor> &out)
    {
        std::vector<cv::RotatedRect> bars;
        std::vector<Pair> pairs;
        std::vector<Armor> quads;
        
        extractBars(mask, bars);
        pairBars(bars, pairs);
        buildQuads(bars, pairs, quads);
        if (params_.mirror.enabled) markReflections(quads);

        for (const Armor &q : quads)
        {
            Armor armor = q;
            armor.is_red = is_red;
            armor.confidence = 1.0f;
            if (armor.mirrored) drawReflection(canvas, armor);
            out.push_back(armor);
        }
    }//启动+传输
    void TvDetector::extractBars(const cv::Mat &mask, std::vector<cv::RotatedRect> &bars) const
    {
        bars.clear();
        const BarFilter &p = params_.bar;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        bars.reserve(contours.size());

        for (size_t i = 0; i < contours.size(); i++)
        {
            const double area = cv::contourArea(contours[i]);
            if (area <= p.min_area) // 去噪
                continue;
            cv::RotatedRect rr = normalizeRect(cv::minAreaRect(contours[i]));
            const float w = rr.size.width, h = rr.size.height;
            if (h < p.min_height) // 太矮
                continue;
            if (h / w < p.min_hw || h / w > p.max_hw) // 不细长
                continue;
            if (w * h > p.max_area) // 超级大灯
                continue;
            if (area / (w * h) < p.min_fill) // 填充率太低
                continue;
            bars.push_back(rr);
        }
    }//找灯条
    bool TvDetector::hasBarBetween(const std::vector<cv::RotatedRect> &bars, int i, int j) const
    {
        cv::Point2f pa[4], pb[4], pc[4];
        bars[i].points(pa);
        bars[j].points(pb);
        cv::Point2f a_top, a_bot, b_top, b_bot;
        sortBarPoints(pa, a_top, a_bot);
        sortBarPoints(pb, b_top, b_bot);
        const std::vector<cv::Point2f> four{a_top, a_bot, b_top, b_bot};
        const cv::Rect box = cv::boundingRect(four);

        for (int k = 0; k < (int)bars.size(); k++)
        {
            if (k == i || k == j)
                continue;
            bars[k].points(pc);
            cv::Point2f k_top, k_bot;
            sortBarPoints(pc, k_top, k_bot);
            if (box.contains(cv::Point(k_top)) || box.contains(cv::Point(k_bot)) ||
                box.contains(cv::Point(bars[k].center)))
                return true;
        }
        return false;
    }//偷腥猫
    void TvDetector::pairBars(const std::vector<cv::RotatedRect> &bars,
                              std::vector<Pair> &pairs) const
    {
        pairs.clear();
        const PairParam &p = params_.pair;

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
                const float h_diff = std::abs(a.size.height - b.size.height) / avg_len;
                if (h_diff > p.h_diff_max)
                    continue;

                // 3. 宽度接近
                const float min_w = std::min(a.size.width, b.size.width);
                const float max_w = std::max(a.size.width, b.size.width);
                if (min_w < 1.0f)
                    continue;
                const float w_ratio = max_w / min_w;
                if (w_ratio > p.w_ratio_max)
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

                // 6. 中间不能夹着第三根灯条
                if (hasBarBetween(bars, i, j))
                    continue;

                const float cost = d_ang * 2.0f + h_diff * 40.0f +
                                   std::max(0.0f, w_ratio - 1.0f) * 6.0f +
                                   dy / max_len * 20.0f;
                pairs.push_back({i, j, cost, ratio >= p.large_ratio});
            }
        }
        std::sort(pairs.begin(), pairs.end(), byCost<Pair>);
    }//配对灯条
    void TvDetector::buildQuads(const std::vector<cv::RotatedRect> &bars,
                                const std::vector<Pair> &pairs,
                                std::vector<Armor> &quads) const
    {
        quads.clear();
        quads.reserve(pairs.size());

        for (const Pair &c : pairs)
        {
            const cv::RotatedRect &bi = bars[c.i];
            const cv::RotatedRect &bj = bars[c.j];
            const bool i_is_left = bi.center.x < bj.center.x;
            const cv::RotatedRect &left_bar = i_is_left ? bi : bj;
            const cv::RotatedRect &right_bar = i_is_left ? bj : bi;

            cv::Point2f left_top, left_bottom, right_top, right_bottom;
            cv::Point2f ptmp[4];
            left_bar.points(ptmp);
            sortBarPoints(ptmp, left_top, left_bottom);
            right_bar.points(ptmp);
            sortBarPoints(ptmp, right_top, right_bottom);

            // 端点对齐：半截灯条 / 错配
            const float h_avg = (left_bar.size.height + right_bar.size.height) * 0.5f;
            const float max_endpoint_dy = params_.pair.endpoint_dy_ratio * h_avg;
            if (std::abs(left_top.y - right_top.y) > max_endpoint_dy)
                continue;
            if (std::abs(left_bottom.y - right_bottom.y) > max_endpoint_dy)
                continue;

            Armor q;
            q.q[0] = left_top;
            q.q[1] = right_top;
            q.q[2] = right_bottom;
            q.q[3] = left_bottom;
            q.center = quadCenter(q.q);
            q.width = quadWidth(q.q);
            q.height = quadHeight(q.q);
            q.bar_left = c.i;
            q.bar_right = c.j;
            q.cost = c.cost;
            q.is_large = c.large;
            q.confidence = 1.0f;
            quads.push_back(q);
        }
    }//找装甲板
    void TvDetector::markReflections(std::vector<Armor> &quads) const
    {
        // 倒影一定在实体下方，所以只把“下方那个”标成倒影，保留上方的真实装甲板
        for (size_t a = 0; a < quads.size(); ++a)
        {
            for (size_t b = 0; b < quads.size(); ++b)
            {
                if (a == b || quads[b].mirrored)
                    continue;
                if (isFloorReflection(quads[a], quads[b]))
                    quads[b].mirrored = true;
            }
        }
    }//处理倒影
    bool TvDetector::isFloorReflection(const Armor &upper, const Armor &lower) const
    {
        const MirrorParam &p = params_.mirror;

        if (upper.width <= 0.f || upper.height <= 0.f ||
            lower.width <= 0.f || lower.height <= 0.f)
            return false;

        // 1. 倒影必须在下方
        if (lower.center.y <= upper.center.y)
            return false;

        const float w_ref = std::max(upper.width, lower.width);

        // 2. 水平位置基本对齐
        if (std::abs(lower.center.x - upper.center.x) > p.dx_ratio * w_ref)
            return false;

        // 3. 宽度接近，容忍倒影被压缩
        const float wr = lower.width / upper.width;
        if (wr < 1.f / p.w_ratio || wr > p.w_ratio)
            return false;

        // 4. 竖直间隔合理：不重叠，也不能远到不像同一块板的倒影
        const float gap = lower.center.y - upper.center.y;
        if (gap < p.min_gap_ratio * upper.height)
            return false;
        if (gap > p.max_gap_ratio * upper.height)
            return false;

        // 5. 上下翻转对应：只比较 x，避免被竖直压缩量干扰
        const float xtol = p.x_tol_ratio * w_ref;
        if (std::abs(upper.q[0].x - lower.q[3].x) > xtol)
            return false;
        if (std::abs(upper.q[1].x - lower.q[2].x) > xtol)
            return false;

        // 6. 倒影不应明显比实体高
        if (lower.height > p.max_height_ratio * upper.height)
            return false;

        return true;
    }//判断倒影
    void TvDetector::drawReflection(cv::Mat &canvas, const Armor &armor) const
    {
        // 灰色框 + MIRROR 标记，回放时用于确认判据是否误伤
        for (int m = 0; m < 4; ++m)
            cv::line(canvas, armor.q[m], armor.q[(m + 1) % 4], cv::Scalar(128, 128, 128), 1);
        cv::putText(canvas, "MIRROR",
                    cv::Point((int)armor.q[0].x - 10, (int)armor.q[0].y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(128, 128, 128), 1);
    }//画倒影
} // namespace armor_detector
