import numpy as np
import sys


def create_uniform_coordinates(n: int, u_bound: float = 1):
    """
    在(0,1)x(0,1)的空间中生成 n 个均匀分布的点。

    Args:
        n (int): 点的个数，必须为完全平方数。
        u_bound (float): 坐标的上限。
        
    Returns:
        一个二元组，分别是两个一维数组，对应坐标数组的横纵坐标。
    """
    assert int(np.sqrt(n)) ** 2 == n, "n 必须为完全平方数"
    
    num_edge = int(np.sqrt(n))
    lin = np.linspace(0, u_bound, num_edge, dtype=np.float64)
    coor_x, coor_y = np.meshgrid(lin, lin)
    coor_x = coor_x.ravel()
    coor_y = coor_y.ravel()
    return coor_x, coor_y

def Move_coor(xc0, yc0, u0, v0, delta_t=0.1, u_bound=1):
    """计算给定速度下，粒子在一定时间间隔后的位置

    Args:
        xc0 (array_like): 粒子的初始横坐标
        yc0 (array_like): 粒子的初始纵坐标
        u0 (array_like): 粒子的横向速度
        v0 (array_like): 粒子的纵向速度
        delta_t (float): 粒子移动的时间间隔,默认为0.1
        u_bound (float): 空间边界的范围,默认为1

    Returns:
        Tuple[array_like, array_like]: 粒子在移动后的横纵坐标

    Notes:
        使用了周期性边界条件(PBC)
    """

    # 计算粒子在x轴和y轴上的移动距离
    dis_x = u0 * delta_t
    dis_y = v0 * delta_t

    # 计算新的粒子横坐标，并对越界的粒子进行处理
    nums = len(xc0)
    xc_new = xc0 + dis_x
    xc_new = np.where(xc_new < 0, xc_new + u_bound, np.where(xc_new > u_bound, xc_new - u_bound, xc_new))

    # 计算新的粒子纵坐标，并对越界的粒子进行处理
    yc_new = yc0 + dis_y
    yc_new = np.where(yc_new < 0, yc_new + u_bound, np.where(yc_new > u_bound, yc_new - u_bound, yc_new))

    # 返回粒子移动后的位置
    return xc_new, yc_new

def Layer(ne, weight=10**4):
    """根据距离边缘的长度分配权重
    
    Args:
        ne:归一化后的距离最近边缘的距离
        weigth:最边缘粒子的权重
    
    Returns:
        weight:各边缘粒子的权重
    """
    
    if ne > 0.08:
        eg_weight = 0.2 * weight
    elif ne >= 0.06:
        eg_weight = 0.4 * weight
    elif ne >= 0.04:
        eg_weight = 0.6 * weight
    elif ne >= 0.02:
        eg_weight = 0.8 * weight
    else:
        eg_weight = 1.0 * weight

    return eg_weight


def Edge_identification(coor_x, coor_y, u_bound=1, per=0.1, weight=10**4):
    """
    根据给定的点坐标集，识别边界和内部的点，并分配边缘点的权重

    Args:
        coor_x (ndarray): 点的 x 坐标集
        coor_y (ndarray): 点的 y 坐标集
        u_bound (float, optional): 边界的宽度,默认为1
        per (float, optional): x 或 y 方向最靠近边缘的百分之几被认为是边界,默认为0.1
        weight (float, optional): 最靠近边缘的粒子的权重,默认为10^4

    Returns:
        tuple: 内部或者边缘的点的序号集合和不同位置的边缘粒子分配的权重

    """
    # 获取点坐标数量
    times = len(coor_x)
    # 初始化内部和边缘点的序号集合以及边缘粒子的权重
    no_boundary = []
    no_inside = []
    weight_edge = []

    # 计算每个点到边界的最小距离
    ll = coor_x / u_bound
    rl = u_bound - coor_x
    dl = coor_y / u_bound
    ul = u_bound - coor_y

    for i in range(times):
        # 获取该点到边界的距离
        minl = min(ll[i], rl[i], dl[i], ul[i])
        ne = minl / u_bound
        if ne <= per:
            # 如果距离小于等于 per，认为该点是边缘点
            no_boundary.append(i)
            # 根据距离边缘的长度分配权重
            weight_edge.append(Layer(ne, weight))
        else:
            # 否则该点是内部点
            no_inside.append(i)

    return no_inside, no_boundary, weight_edge

























