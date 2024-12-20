import numpy as np
import matplotlib.pyplot as plt

# 设置字体样式以正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义距离变换类
class DistanceTransform:
    # 初始化距离变换类
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape

    # 定义欧氏距离变换
    def Euclidean_distance_transform(self):
        distance_matrix = np.zeros_like(self.image, dtype=float)
        # 获取前景像素点的坐标
        yy, xx = np.argwhere(self.image == 1).T
        for y in range(self.height):
            for x in range(self.width):
                if self.image[y, x] == 1:
                    distance_matrix[y, x] = 0
                else:
                    # 计算当前像素到前景像素点的欧氏距离
                    distances = np.sqrt((y - yy) ** 2 + (x - xx) ** 2)
                    # 取最小值作为当前像素的距离值
                    distance_matrix[y, x] = np.min(distances)
        return distance_matrix

    # 定义城市街区距离变换
    def D4_distance_transform(self):
        distance_matrix = np.zeros_like(self.image, dtype=float)
        yy, xx = np.argwhere(self.image == 1).T
        for y in range(self.height):
            for x in range(self.width):
                if self.image[y, x] == 1:
                    distance_matrix[y, x] = 0
                else:
                    # 计算当前像素到前景像素点的曼哈顿距离
                    distances = np.abs(y - yy) + np.abs(x - xx)
                    # 取最小值作为当前像素的距离值
                    distance_matrix[y, x] = np.min(distances)
        return distance_matrix

    # 定义棋盘距离变换
    def D8_distance_transform(self):
        distance_matrix = np.zeros_like(self.image, dtype=float)
        yy, xx = np.argwhere(self.image == 1).T
        for y in range(self.height):
            for x in range(self.width):
                if self.image[y, x] == 1:
                    distance_matrix[y, x] = 0
                else:
                    # 计算当前像素到前景像素点的棋盘距离
                    distances = np.maximum(np.abs(y - yy), np.abs(x - xx))
                    # 取最小值作为当前像素的距离值
                    distance_matrix[y, x] = np.min(distances)
        return distance_matrix


# 初始化输入图像：480x480的全黑背景
image = np.zeros((480, 480), dtype=np.uint8)
# 取三个前景像素点
image[100, 200] = 1
image[200, 100] = 1
image[300, 300] = 1

# 显示原始图像
plt.figure(figsize=(5, 5))
plt.scatter([100, 200, 300], [200, 100, 300], color='white', marker='o')
plt.imshow(image, cmap='gray')
plt.title('原始图像', fontsize=15)

# 计算距离变换矩阵
dt = DistanceTransform(image)
euclidean_distance_matrix = dt.Euclidean_distance_transform()
manhattan_distance_matrix = dt.D4_distance_transform()
chessboard_distance_matrix = dt.D8_distance_transform()

# 输出欧氏、城区和棋盘的距离矩阵
print("欧氏距离的变换矩阵:\n", euclidean_distance_matrix)
print("城区距离的变换矩阵:\n", manhattan_distance_matrix)
print("棋盘距离的变换矩阵:\n", chessboard_distance_matrix)

# 可视化距离变换结果
plt.figure(figsize=(15, 5))

# 欧氏距离变换
plt.subplot(1, 3, 1)
plt.imshow(euclidean_distance_matrix, cmap='gray')
plt.colorbar(shrink=0.8)
plt.title('欧氏距离', fontsize=15)
plt.axis('off')

# 城区距离变换
plt.subplot(1, 3, 2)
plt.imshow(manhattan_distance_matrix, cmap='gray')
plt.colorbar(shrink=0.8)
plt.title('城区距离', fontsize=15)
plt.axis('off')

# 棋盘距离变换
plt.subplot(1, 3, 3)
plt.imshow(chessboard_distance_matrix, cmap='gray')
plt.colorbar(shrink=0.8)
plt.title('棋盘距离', fontsize=15)
plt.axis('off')

plt.tight_layout()
plt.show()
