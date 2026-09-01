import numpy as np

# 1.1 从列表创建数组
print("=== 从列表创建 ===")
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)
print("数据类型:", arr1.dtype)
print("形状:", arr1.shape)

# 1.2 创建二维数组（矩阵）
print("\n=== 创建二维数组 ===")
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2)
print("形状:", arr2.shape)  # (3, 3) 表示3行3列

# 1.3 用 np.random.randn 创建随机数组（重点！）
print("\n=== 用 randn 创建随机数组 ===")
# randn 生成标准正态分布（均值为0，标准差为1）的随机数
random_arr = np.random.randn(3, 4)  # 3行4列的随机矩阵
print(random_arr)
print("均值:", random_arr.mean())  # 应该接近0
print("标准差:", random_arr.std())  # 应该接近1

# 1.4 其他常用的创建方式
print("\n=== 其他创建方式 ===")
zeros = np.zeros((2, 3))      # 全0矩阵
ones = np.ones((2, 3))        # 全1矩阵
eye = np.eye(3)               # 单位矩阵（对角线为1）
arange = np.arange(10)        # 类似Python的range，生成0-9
linspace = np.linspace(0, 1, 5)  # 0到1之间均匀取5个数

print("zeros:\n", zeros)
print("ones:\n", ones)
print("eye:\n", eye)
print("arange:", arange)
print("linspace:", linspace)

# 创建一个3行4列的随机数组
arr = np.random.randn(3, 4)
print("原始数组:")
print(arr)

# 2.1 一维切片（和列表一样）
print("\n=== 一维切片 ===")
print("第0行:", arr[0])          # 取第0行（索引从0开始）
print("第1行第2列:", arr[1, 2])  # 取第1行第2列的元素
print("第2行最后两列:", arr[2, -2:])  # 冒号表示切片

# 2.2 二维切片（行切片和列切片）
print("\n=== 二维切片 ===")
print("前两行:\n", arr[:2, :])    # 取前两行所有列
print("前两列:\n", arr[:, :2])    # 取所有行的前两列
print("第1-2行，第1-2列:\n", arr[1:3, 1:3])  # 取一个子矩阵

# 2.3 花式索引（用列表取不连续的行/列）
print("\n=== 花式索引 ===")
print("取第0行和第2行:\n", arr[[0, 2], :])  # 取第0行和第2行
print("取第1列和第3列:\n", arr[:, [1, 3]])  # 取第1列和第3列

# 2.4 布尔索引（条件筛选）
print("\n=== 布尔索引 ===")
mask = arr > 0  # 生成一个布尔矩阵，标记大于0的位置
print("大于0的位置:\n", mask)
print("所有大于0的元素:", arr[arr > 0])  # 取出所有大于0的元素
print("大于0的元素个数:", np.sum(arr > 0))