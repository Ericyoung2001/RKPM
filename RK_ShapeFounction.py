import numpy as np
from CoorThing import create_uniform_coordinates
from Kernel import Kernel


def Shape_matrix(coor_x, coor_y, nc=169, ns=36, SideLen=1):
    """计算形函数矩阵 (对应二阶PDE)"""
    ## xc和yc为配点的坐标
    ## nc和ns为配点数目和源点数目
    ## SideLen为空间边界长度
    ## 输出为(nc,ns)的型函数矩阵及其导数
    shp = np.zeros(shape=(nc,ns), dtype=np.float64)
    shpdx = np.zeros(shape=(nc,ns), dtype=np.float64)
    shpdy = np.zeros(shape=(nc,ns), dtype=np.float64)
    shpdxx = np.zeros(shape=(nc,ns), dtype=np.float64)
    shpdxy = np.zeros(shape=(nc,ns), dtype=np.float64)
    shpdyy = np.zeros(shape=(nc,ns), dtype=np.float64)
    
    xc_all, yc_all = coor_x, coor_y 
    xs_all, ys_all = create_uniform_coordinates(ns, SideLen)
    dx = SideLen/(np.sqrt(ns)+1)
    dy = SideLen/(np.sqrt(ns)+1)

    for i in range(nc):
        xc = xc_all[i]
        yc = yc_all[i]
        H0 = np.array([1,0,0,0,0,0]).reshape(-1,1)
        M = np.zeros(shape=(6,6), dtype=np.float64)
        Mdx = np.zeros(shape=(6,6), dtype=np.float64)
        Mdy = np.zeros(shape=(6,6), dtype=np.float64)
        Mdxx = np.zeros(shape=(6,6), dtype=np.float64)
        Mdxy = np.zeros(shape=(6,6), dtype=np.float64)
        Mdyy = np.zeros(shape=(6,6), dtype=np.float64)
        
        for j in range(ns):
            xs = xs_all[j]
            ys = ys_all[j]
            w, wdx, wdy, wdxx, wdxy, wdyy = Kernel(xc,yc,xs,ys,dx,dy)
            Ht    = np.array([1, xc-xs, yc-ys, (xc-xs)**2, (xc-xs)*(yc-ys), (yc-ys)**2]).reshape(-1,1)
            Htdx  = np.array([0,     1,     0,  2*(xc-ys),         (yc-ys),          0]).reshape(-1,1)
            Htdy  = np.array([0,     0,     1,          0, (xc-ys)        ,  2*(yc-ys)]).reshape(-1,1)
            Htdxx = np.array([0,     0,     0,          2,               0,          0]).reshape(-1,1)
            Htdxy = np.array([0,     0,     0,          0,               1,          0]).reshape(-1,1)
            Htdyy = np.array([0,     0,     0,          0,               0,          2]).reshape(-1,1)
            M += np.matmul(Ht,Ht.T)*w
            Mdx += np.matmul(Htdx,Ht.T)*w + np.matmul(Ht,Htdx.T)*w + np.matmul(Ht,Ht.T)*wdx
            Mdy += np.matmul(Htdy,Ht.T)*w + np.matmul(Ht,Htdy.T)*w + np.matmul(Ht,Ht.T)*wdy
            Mdxx += np.matmul(Htdxx,Ht.T)*w + np.matmul(Ht,Htdxx.T)*w + np.matmul(Ht,Ht.T)*wdxx \
                    + 2*(np.matmul(Htdx,Htdx.T)*w + np.matmul(Ht,Htdx.T)*wdx + np.matmul(Htdx,Ht.T)*wdx)
            Mdyy += np.matmul(Htdyy,Ht.T)*w + np.matmul(Ht,Htdyy.T)*w + np.matmul(Ht,Ht.T)*wdyy \
                    + 2*(np.matmul(Htdy,Htdy.T)*w + np.matmul(Ht,Htdy.T)*wdy + np.matmul(Htdy,Ht.T)*wdy)
            Mdxy += np.matmul(Htdxy,Ht.T)*w + np.matmul(Htdx,Htdy.T)*w + np.matmul(Htdx,Ht.T)*wdy \
                    + np.matmul(Htdy,Htdx.T)*w + np.matmul(Ht,Htdxy.T)*w + np.matmul(Ht,Htdx.T)*wdy \
                    + np.matmul(Htdy,Ht.T)*wdx + np.matmul(Ht,Htdy.T)*wdx + np.matmul(Ht,Ht.T)*wdxy
        Minv = np.linalg.pinv(M)
        Minvdx = - np.matmul(np.matmul(Minv,Mdx),Minv)
        Minvdy = - np.matmul(np.matmul(Minv,Mdy),Minv)
        Minvdxx = np.matmul(np.matmul(Minv,(2*np.matmul(np.matmul(Mdx,Minv),Mdx)-Mdxx)),Minv)
        Minvdyy = np.matmul(np.matmul(Minv,(2*np.matmul(np.matmul(Mdy,Minv),Mdy)-Mdyy)),Minv)
        Minvdxy = np.matmul(np.matmul(Minv,(np.matmul(np.matmul(Mdy,Minv),Mdx)+np.matmul(np.matmul(Mdx,Minv),Mdy)-Mdxy)),Minv)
        for j in range(ns):
            xs = xs_all[j]
            ys = ys_all[j]
            w, wdx, wdy, wdxx, wdxy, wdyy = Kernel(xc,yc,xs,ys,dx,dy)
            H    = np.array([1, xc-xs, yc-ys, (xc-xs)**2, (xc-xs)*(yc-ys), (yc-ys)**2]).reshape(-1,1)
            Hdx  = np.array([0,     1,     0,  2*(xc-ys),         (yc-ys),          0]).reshape(-1,1)
            Hdy  = np.array([0,     0,     1,          0, (xc-ys)        ,  2*(yc-ys)]).reshape(-1,1)
            Hdxx = np.array([0,     0,     0,          2,               0,          0]).reshape(-1,1)
            Hdxy = np.array([0,     0,     0,          0,               1,          0]).reshape(-1,1)
            Hdyy = np.array([0,     0,     0,          0,               0,          2]).reshape(-1,1)

            shp[i][j] = np.matmul(np.matmul(H0.T,Minv),H)*w
            shpdx[i][j] = np.matmul(H0.T,(np.matmul(Minvdx,H)*w +np.matmul(Minv,Hdx)*w +np.matmul(Minv,H)*wdx))
            shpdy[i][j] = np.matmul(H0.T,(np.matmul(Minvdy,H)*w +np.matmul(Minv,Hdy)*w +np.matmul(Minv,H)*wdy))
            shpdxx[i][j] = np.matmul(H0.T,( np.matmul(Minvdx*2,(Hdx*w + H*wdx)) \
                           + np.matmul(Minv*2,Hdx)*wdx + np.matmul(Minvdxx,H)*w  \
                           + np.matmul(Minv,(Hdxx*w + H*wdxx))))
            shpdyy[i][j] = np.matmul(H0.T,( np.matmul(Minvdy*2,(Hdy*w + H*wdy)) \
                           + np.matmul(Minv*2,Hdy)*wdy + np.matmul(Minvdyy,H)*w  \
                           + np.matmul(Minv,(Hdyy*w + H*wdyy))))
            shpdxy[i][j] = np.matmul(H0.T,(np.matmul(Minvdy,(Hdx*w + H*wdx)) \
                           + np.matmul(Minvdx,(Hdy*w + H*wdy)) \
                           + np.matmul(Minv,(Hdxy*w+Hdx*wdy+Hdy*wdx+H*wdxy)) \
                           + np.matmul(Minvdxy,H)*w))

    return shp, shpdx, shpdy, shpdxx, shpdxy, shpdyy


xc, yc=create_uniform_coordinates(169)
shp, shpdx, shpdxx, shpdxy, shpdyy = Shape_matrix(xc,yc)
print(shp.shape)






