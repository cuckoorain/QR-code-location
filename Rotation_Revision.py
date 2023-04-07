import numpy as np

# 三角定位
class Triangle:
    def __init__(self, PointsList, ideal_length):
        # points 是某一个定位点正方形的[x,y,w,h]
        # PointsList是三个点按顺序（左上，右上，左下）根据[x,y,w,h]得到的点的集合，len==3

        # ideal_length指的是三个定位点正中间构成的理想等腰直角三角形的两条腰长，但是矛盾的是
        # 这个理想的值是旋转修正后得到的像素值，但是我们修正的目的就是求更精准的这个值，所以这个ideal_length可以通过如下方法确定
        # 对于一个端端正正的二维码，三个定位点都是正方形，边长为a，二维码整体作为一个正方形边长为b，则有b/a为一个常数c，
        # 我们只需要找到给定的四组，每组四个定位点构成的四边形最大的边作为a_hat，就可以得到ideal_length = a_hat * c
        # 这个理论在平行光假设中是不成立的，因为旋转加投影不会改变同一直线上线段之间的比例关系
        # 但是在极端情况或是实际情况下是有用的，因为当二维码足够近的时候，在二维码纸面上已经呈现明显的透视效果，
        # 而透视并不遵守这个对于线段的等比定理，所以这个时候，我们这个方法是有其作用的。
        # 且注意，这个计算出来的ideal_length并不一定是第0点对应的定位点，因为可能是右上角或左下角的点更接近摄像头，进而被选中进行计算
        # 那么有两个策略：
        #   1.直接假设这个是第0点对应的边，这样做非常方便，但是似乎是错误的
        #   2.提出另外两组方程组，根据被选中的点带入对应方程组解出答案
        #       2.1 这两组方程的输入如下：（分别为第0点，第1点，第2点）
        #       2.1.1 以第0点为原点(下面的方程组就是以这个为原型的)
        #       [0]         [a]     [a]
        #       [0]         [0]     [-a]
        #       [0]         [0]     [0]
        #       2.1.2 以第1点为原点
        #       [-a]        [0]     [-a]
        #       [0]         [0]     [-a]
        #       [0]         [0]     [0]
        #       2.1.1 以第2点为原点
        #       [0]         [a]     [0]
        #       [-a]        [-a]    [0]
        #       [0]         [0]     [0]

        self.ideal_length = ideal_length
        self.PointsList = PointsList
        self.Triangle_Points = []
        self.Triangle_Points = self.Calculate_Triangle_Points(self)

    def Calculate_Triangle_Points(self):
        for points in self.PointsList:
            x = points[0] + 1 / 2 * points[2]
            y = points[1] + 1 / 2 * points[3]
            self.Triangle_Points.append([x, y])
        return self.Triangle_Points

    # 理论推导如下
    # 修正可以做很多文章，一个等腰直角三角形与一个不共面但是共直角顶点的随机三角形在三维空间中的映射似乎是唯一的，似乎可以这样做：
    # 设零点为直角点（第0点），(a,0,0)和(0,-a,0)为其余两个点（第1点和第2点），三点构成等腰直角三角形
    # 然后设整个空间绕x轴转动alpha角度，使得xoy旋转后的平面与原来zoy平面的交线与y正方向夹角为alpha，则有矩阵
    # (第0点)
    ###
    # [1,   0,                      0 ]                  [0]    [0]
    # [0,   cos(alpha),    -sin(alpha)]          *       [0] =  [0]
    # [0,   sin(alpha),     cos(alpha)]                  [0]    [0]

    # (第1点)
    ###
    # [1,   0,                      0 ]                  [a]    [a]
    # [0,   cos(alpha),    -sin(alpha)]          *       [0] =  [0]
    # [0,   sin(alpha),     cos(alpha)]                  [0]    [0]

    # (第2点)
    ###
    # [1,   0,                      0 ]                  [0]     [0]
    # [0,   cos(alpha),    -sin(alpha)]          *       [-a] =  [-a*cos(alpha)]
    # [0,   sin(alpha),     cos(alpha)]                  [0]     [-a*sin(alpha)]

    # 然后设整个空间绕y轴转动beta角度，使得xoy旋转后的平面与原来zox平面的交线与x正方向夹角为beta，则有矩阵
    # (第0点)
    ###
    # [cos(beta),   0,      -sin(beta)]                  [0]                    [0]
    # [0,           1,              0 ]          *       [0]             =      [0]
    # [sin(beta),   0,      cos(beta) ]                  [0]                    [0]

    # (第1点)
    ###
    # [cos(beta),   0,      -sin(beta)]                  [a]                    [a*cos(beta)]
    # [0,           1,              0 ]  *               [0]             =      [0]
    # [sin(beta),   0,      cos(beta) ]                  [0]                    [a*sin(beta)]

    # (第2点)
    ###
    # [cos(beta),   0,      -sin(beta)]                  [0]                    [0]
    # [0,           1,              0 ]          *       [-a*cos(alpha)]  =     [-a*cos(alpha)]
    # [sin(beta),   0,      cos(beta) ]                  [-a*sin(alpha)]        [-a*sin(alpha)*cos(beta)]

    # 然后设整个空间绕z轴转动gamma角度，使得zoy旋转后的平面与原来yox平面的交线与y正方向夹角为gamma，则有矩阵
    # (第0点)
    ###
    # [-cos(gamma),   sin(gamma),     0]             [0]                                [0]
    # [-sin(gamma),   cos(gamma),     0]        *    [0]                        =       [0]
    # [0,             0,              1]             [0]                                [0]

    # (第1点)
    ###
    # [-cos(gamma),   sin(gamma),     0]             [a*cos(beta)]                      [-a*cos(beta)*cos(gamma)]
    # [-sin(gamma),   cos(gamma),     0]     *       [0]                        =       [-a*cos(beta)*sin(gamma)]
    # [0,             0,              1]             [a*sin(beta)]                      [a*sin(beta)]

    # (第2点)
    ###
    # [-cos(gamma),   sin(gamma),     0]             [0]                                [-a*cos(alpha)*sin(gamma)]
    # [-sin(gamma),   cos(gamma),     0]          *  [-a*cos(alpha)]             =      [-a*cos(alpha)*cos(gamma)]
    # [0,             0,              1]             [-a*sin(alpha)*cos(beta)]          [-a*sin(alpha)*cos(beta) ]

    # 这样的话只用解方程，设任意三角形的三点为(0,0,0),(x1,y1,z1),(x2,y2,z2)
    # 对应方程可得
    # x1 = [-a*cos(beta)*cos(gamma)]
    # y1 = [-a*cos(beta)*sin(gamma)]
    # z1 = [a*sin(beta)]
    # x2 = [-a*cos(alpha)*sin(gamma)]
    # y2 = [-a*cos(alpha)*cos(gamma)]
    # z2 = [-a*sin(alpha)*cos(beta) ]

    # 又由于假设点光源与纸面的距离远大于和纸面和屏幕之间的距离，则可以视为平行光，
    # （其中纸面和屏幕并不平行，理想状态下，纸面应该为等腰直角三角形，但现实为随机三角形）
    # 进而在随机三角形的伪直角点处作一个平面平行于屏幕，则为屏幕实际的投影，即函数中接收到的三个点
    # 所以这三个点并没有z坐标传输到函数中（认为屏幕到纸面为z轴正方向，point0到point1方向为x轴正方向，这样的y轴方向和程序一致）
    # 所以实际方程为：
    # x1 = [-a*cos(beta)*cos(gamma)]
    # y1 = [-a*cos(beta)*sin(gamma)]
    # x2 = [-a*cos(alpha)*sin(gamma)]
    # y2 = [-a*cos(alpha)*cos(gamma)]
    # 四组方程，四个未知数，非常完美地可解...是吗？
    # 很遗憾，不是的，他们是线性相关的，对于集合意义来说，可以是这样：
    # 从原点出发的两条三维空间矢量A1，B1长度各自固定但角度可变，对于任意角度的A1在xoy的投影aa1，
    # 总有四个B1满足以下条件：即xoy平面上B1的投影bb1与aa1长度相等且互相垂直
    # 所以这个方程组是不完备的，无法解决所有变量
    # 但是令人欣慰的是，我们能解出gamma以及alpha和beta的关系
    # 令人更加欣慰的是，在这个特殊的问题中，a是已知的，所以所有的变量都是可以知道的，虽然这样依赖于a的解法显得非常弱，
    # 但是起码我们能做出来了

    def Rotation_Revision_Point0AsZeroPoint(self):
        # 对于第0点作为原点的三维旋转修正
        # 返回的是弧度制
        zero_point = self.Triangle_Points[0]
        point1 = self.Triangle_Points[1]
        point2 = self.Triangle_Points[2]
        x1 = point1[0] - zero_point[0]
        y1 = point1[1] - zero_point[1]
        x2 = point2[0] - zero_point[0]
        y2 = point2[1] - zero_point[1]

        # 根据理论解方程
        gamma_1 = np.arctan(y1 / x1)
        gamma_2 = np.arctan(y2 / x2)
        gamma_avg = (gamma_2 + gamma_1) / 2
        beta_1 = -x1 / self.ideal_length / np.cos(gamma_avg)
        beta_2 = -y1 / self.ideal_length / np.sin(gamma_avg)
        beta_avg = (beta_2 + beta_1) / 2
        alpha_1 = -x2 / self.ideal_length / np.sin(gamma_avg)
        alpha_2 = -y2 / self.ideal_length / np.cos(gamma_avg)
        alpha_avg = (alpha_2 + alpha_1) / 2
        # 或许误差太大可以加一个报警系统，但是这是论文，所以还是算了

        # 返回的是弧度制
        return [alpha_avg, beta_avg, gamma_avg]

    def Rotation_Revision_Point1AsZeroPoint(self):
        # 需要修改########################################################################################
        # 对于第1点作为原点的三维旋转修正
        # 返回的是弧度制
        zero_point = self.Triangle_Points[0]
        point1 = self.Triangle_Points[1]
        point2 = self.Triangle_Points[2]
        x1 = point1[0] - zero_point[0]
        y1 = point1[1] - zero_point[1]
        x2 = point2[0] - zero_point[0]
        y2 = point2[1] - zero_point[1]

        # 根据理论解方程
        gamma_1 = np.arctan(y1 / x1)
        gamma_2 = np.arctan(y2 / x2)
        gamma_avg = (gamma_2 + gamma_1) / 2
        beta_1 = -x1 / self.ideal_length / np.cos(gamma_avg)
        beta_2 = -y1 / self.ideal_length / np.sin(gamma_avg)
        beta_avg = (beta_2 + beta_1) / 2
        alpha_1 = -x2 / self.ideal_length / np.sin(gamma_avg)
        alpha_2 = -y2 / self.ideal_length / np.cos(gamma_avg)
        alpha_avg = (alpha_2 + alpha_1) / 2
        # 或许误差太大可以加一个报警系统，但是这是论文，所以还是算了

        # 返回的是弧度制
        return [alpha_avg, beta_avg, gamma_avg]

    def Rotation_Revision_Point2AsZeroPoint(self):
        # 对于第2点作为原点的三维旋转修正
        # 需要修改########################################################################################
        # 返回的是弧度制
        zero_point = self.Triangle_Points[0]
        point1 = self.Triangle_Points[1]
        point2 = self.Triangle_Points[2]
        x1 = point1[0] - zero_point[0]
        y1 = point1[1] - zero_point[1]
        x2 = point2[0] - zero_point[0]
        y2 = point2[1] - zero_point[1]

        # 根据理论解方程
        gamma_1 = np.arctan(y1 / x1)
        gamma_2 = np.arctan(y2 / x2)
        gamma_avg = (gamma_2 + gamma_1) / 2
        beta_1 = -x1 / self.ideal_length / np.cos(gamma_avg)
        beta_2 = -y1 / self.ideal_length / np.sin(gamma_avg)
        beta_avg = (beta_2 + beta_1) / 2
        alpha_1 = -x2 / self.ideal_length / np.sin(gamma_avg)
        alpha_2 = -y2 / self.ideal_length / np.cos(gamma_avg)
        alpha_avg = (alpha_2 + alpha_1) / 2
        # 或许误差太大可以加一个报警系统，但是这是论文，所以还是算了

        # 返回的是弧度制
        return [alpha_avg, beta_avg, gamma_avg]