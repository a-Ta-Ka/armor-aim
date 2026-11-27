#pragma once

#include <opencv2/core.hpp>

namespace armor_detector
{
    // 一块装甲板：两种前端都填这个结构，后面的 PnP、TF、发消息只写一份
    struct Armor
    {
        cv::Point2f q[4];      // 左上、右上、右下、左下
        cv::Point2f center;    // 四角点平均
        float width = 0.f;     // 上下边平均长度（像素）
        float height = 0.f;    // 左右边平均长度（像素）

        bool is_red = false;
        bool is_large = false;

        int number = -1;         // 编号 0~8；传统视觉没有编号，填 -1
        float confidence = 0.f;  // 神经网络分数；传统视觉填 1.0

        bool mirrored = false;   // 仅传统视觉使用：是否被判为地面倒影

        // 以下仅传统视觉内部使用
        int bar_left = -1;       // 左灯条在 bars 里的下标
        int bar_right = -1;      // 右灯条在 bars 里的下标
        float cost = 0.f;        // 配对代价，越小越优先
    };

    // 灯条筛选
    struct BarFilter
    {
        int min_area = 25;         // 轮廓面积下限
        int min_height = 6;        // 灯条高度下限（像素）
        float min_hw = 2.0f;       // 长宽比下限 h/w
        float max_hw = 30.0f;      // 长宽比上限
        float max_area = 20000.0f; // 面积上限（排除装甲板底色块）
        float min_fill = 0.1f;     // 填充率
    };

    // 灯条配对
    struct PairParam
    {
        float dy_ratio = 0.25f;     // |dy| <= dy_ratio * max_len
        float h_diff_max = 0.15f;   // |Δh| / 平均灯条高
        float w_ratio_max = 2.5f;   // 宽灯条/窄灯条 上限
        float ang_diff_max = 8.0f;  // 角度差（度）
        float wl_min = 1.2f;        // 中心距 / 平均灯条长  下限
        float wl_max = 6.0f;        // 中心距 / 平均灯条长  上限
        float large_ratio = 2.9f;   // >= 该值判为大装甲板（onInit 里按 wl_large/wl_small 重算）
        float endpoint_dy_ratio = 0.35f; // 上下端点允许的纵向差 / 平均灯条长度
    };

    // 地面倒影判据
    struct MirrorParam
    {
        bool enabled = true;
        float dx_ratio = 0.35f;         // 水平偏移 / 平均宽度
        float w_ratio = 1.6f;           // 宽度比上限（倒影可能被压缩）
        float x_tol_ratio = 0.25f;      // 角点 x 对应容差 / 平均宽度
        float min_gap_ratio = 0.80f;    // 竖直间隔下限 / 实体高度
        float max_gap_ratio = 4.00f;    // 竖直间隔上限 / 实体高度
        float max_height_ratio = 1.20f; // 倒影高度 / 实体高度
    };

    // 六个 HSV 区间
    struct HsvRange
    {
        int bh_min = 80, bh_max = 120;
        int rh_min = 0, rh_max = 40;
        int rh2_min = 170, rh2_max = 179;
        int s_min = 80, s_max = 255;
        int v_min = 160, v_max = 255;
    };
  
    // 单块装甲板观测结果
    struct Observation
    {
        cv::Point3d p;        // 装甲板中心，相机坐标系
        cv::Point3d n;        // 装甲板 z 轴朝向：由装甲板指向车体中心
        cv::Point3d c;        // 车体中心观测值 = p + radius * n
        double alpha = 0.0;   // 该装甲板的相位角/角度
        double face_on = 0.0; // 正对程度
        int det_id = 0;       // 检测结果 ID
        bool red = false;
    };

} // namespace armor_detector
