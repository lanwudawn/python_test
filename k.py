import numpy as np
import matplotlib.pyplot as plt
# 数据准备
X = np.array([
    [2, 3],
    [3, 3],
    [6, 5],
    [8, 8],
])
# 定义欧式距离计算函数
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
# K-means算法实现
def k_means(X, k, max_iters=100):
    # 随机选择K个初始簇中心
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        # 分配数据点到最近的簇
        labels = []
        for point in X:
            distances = [euclidean_distance(point, center) for center in centers]
            labels.append(np.argmin(distances))
        labels = np.array(labels)
        # 更新簇中心
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # 检查是否收敛
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels
# 测试代码
if __name__ == "__main__":
    # 测试欧式距离
    a = np.array([1, 2])
    b = np.array([4, 6])
    print("欧式距离:", euclidean_distance(a, b))  # 应输出5.0
    # K-means聚类测试
    k = 2
    centers, labels = k_means(X, k)
    print("\n最终的簇中心：")
    print(centers)
    print("\n每个数据点的簇标签：")
    print(labels)
    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()