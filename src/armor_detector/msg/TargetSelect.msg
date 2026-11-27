std_msgs/Header header

bool valid       # 有无真实数据
bool predicted   # 是否推测出来
uint32 target_id

geometry_msgs/Point position                     # 当前选中装甲板中心
geometry_msgs/Point position_predict             # 预测同一块装甲板T秒后中心
geometry_msgs/Point position_baseline_predict    # 同样 T 秒后的基础 position+velocity*T 预测，用于对比
geometry_msgs/Point car_center                   # 车体中心估计

float32 phase    # 车体相位
float32 omega    # 车体角速度

float32 radius     # 车心到装甲板距离
uint8 face_index   # 当前板号
uint8 plate_count  # 装甲板数量
