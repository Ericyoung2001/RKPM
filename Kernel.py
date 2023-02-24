import numpy as np


# def CubicB_Spline(r, a):
#     """三次B样条"""

#     ## 输入为配点源点的坐标差值和源点间距
#     z = np.abs(r)/a
#     if z < 1:
#         phi = 2/3-4*z**2+4*z**3
#         phid = (-8*z+12*z**2)*(np.sign(r)/a)
#         phidd = (-8+24*z)*(np.sign(r)/a)**2
#     elif z < 2:
#         phi = 4/3-4*z+4*z**2-4/3*z**3
#         phid = (-4+8*z-4*z**2)*(np.sign(r)/a)
#         phidd = (8-8*z)*(np.sign(r)/a)**2
#     else:
#         phi = 0
#         phid = 0
#         phidd = 0
    
#     return phi, phid, phidd

# def Kernel(xc, yc, xs, ys, dx, dy):
#     """返回对应的核函数及其导数值"""
#     ## 输入配点及源点坐标，源点间距
#     rx = xc - xs
#     ry = yc - ys
#     phix, phidx, phiddx = CubicB_Spline(rx, dx)
#     phiy, phidy, phidyy = CubicB_Spline(ry, dy)
#     w = phix * phiy
#     wdx = phidx * phiy
#     wdy = phix * phidy
#     wdxx = phiddx * phiy
#     wdxy = phidx * phidy
#     wdyy = phix * phidyy


#     return w, wdx, wdy, wdxx, wdxy, wdyy

# import numpy as np

def CubicB_Spline(r, a):
    """
    三次B样条

    Args:
        r: 配点源点的坐标差值
        a: 源点间距

    Returns:
        phi: B样条函数的值
        phid: B样条函数的导数值
        phidd: B样条函数的二阶导数值
    """
    # 计算z
    z = np.abs(r) / a

    # 根据z的大小，计算phi、phid和phidd
    if z < 1:
        phi = 2 / 3 - 4 * z ** 2 + 4 * z ** 3
        phid = (-8 * z + 12 * z ** 2) * (np.sign(r) / a)
        phidd = (-8 + 24 * z) * (np.sign(r) / a) ** 2
    elif z < 2:
        phi = 4 / 3 - 4 * z + 4 * z ** 2 - 4 / 3 * z ** 3
        phid = (-4 + 8 * z - 4 * z ** 2) * (np.sign(r) / a)
        phidd = (8 - 8 * z) * (np.sign(r) / a) ** 2
    else:
        phi = 0
        phid = 0
        phidd = 0
    
    return phi, phid, phidd


def Kernel(xc, yc, xs, ys, dx, dy):
    """
    返回对应的核函数及其导数值

    Args:
        xc: 配点坐标 x
        yc: 配点坐标 y
        xs: 源点坐标 x
        ys: 源点坐标 y
        dx: 源点间距 x
        dy: 源点间距 y

    Returns:
        w: 核函数的值
        wdx: 核函数对 x 的导数值
        wdy: 核函数对 y 的导数值
        wdxx: 核函数对 x 的二阶导数值
        wdxy: 核函数对 x,y 的混合导数值
        wdyy: 核函数对 y 的二阶导数值
    """
    # 计算 rx 和 ry
    rx = xc - xs
    ry = yc - ys

    # 计算 B 样条函数及其导数值
    phix, phidx, phiddx = CubicB_Spline(rx, dx)
    phiy, phidy, phidyy = CubicB_Spline(ry, dy)

    # 计算核函数及其导数值
    w = phix * phiy
    wdx = phidx * phiy
    wdy = phix * phidy
    wdxx = phiddx * phiy
    wdxy = phidx * phidy
    wdyy = phix * phidyy

    return w, wdx, wdy, wdxx, wdxy, wdyy







