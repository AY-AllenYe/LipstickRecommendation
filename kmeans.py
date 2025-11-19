import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

input_file = 'datasets/lipstick.csv'
output_file = 'datasets/lipstick_clusters.csv'
cluster_dict_file = 'datasets/dict.csv'

# 读取数据
df = pd.read_csv(input_file)
print(f"数据集大小: {df.shape}")
print(df.head())

# 特征选择 - 使用多种颜色空间特征
features = ['R', 'G', 'B', 'H', 'S', 'V']
X = df[features].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("特征数据形状:", X_scaled.shape)

def find_optimal_k(X, max_k=15):
    """
    使用肘部法则和轮廓系数自动确定最佳K值
    """
    wcss = []  # 簇内平方和
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        # 计算轮廓系数
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
    
    return wcss, silhouette_scores, k_range

def plot_optimal_k(wcss, silhouette_scores, k_range):
    """
    绘制确定最佳K值的图表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 肘部法则图
    ax1.plot(k_range, wcss, 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Elbow Method for Optimal K')
    ax1.grid(True)
    
    # 轮廓系数图 - 修复维度问题
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal K')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 自动确定最佳K值
wcss, silhouette_scores, k_range = find_optimal_k(X_scaled)
plot_optimal_k(wcss, silhouette_scores, k_range)

def auto_select_k(silhouette_scores, k_range, threshold=0.002):
    """
    自动选择最佳K值：当轮廓系数增长小于阈值时停止
    """
    # 找到轮廓系数最大的索引
    best_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_idx]
    best_score = silhouette_scores[best_idx]
    
    print(f"轮廓系数最高的K值: K={best_k}, 轮廓系数={best_score:.4f}")
    
    # 显示所有K值的轮廓系数排名
    sorted_indices = np.argsort(silhouette_scores)[::-1]  # 降序排列
    print("\n轮廓系数排名:")
    for i, idx in enumerate(sorted_indices[:5]):  # 显示前5名
        print(f"{i+1}. K={k_range[idx]}: {silhouette_scores[idx]:.4f}")
    
    return best_k

# 自动选择K值
optimal_k = auto_select_k(silhouette_scores, list(k_range))
print(f"自动选择的最佳K值: {optimal_k}")

# 使用最佳K值进行最终聚类
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = final_kmeans.fit_predict(X_scaled)

def analyze_clusters(df, features):
    """
    分析聚类结果
    """
    cluster_summary = df.groupby('cluster').agg({
        'R': ['mean', 'std'],
        'G': ['mean', 'std'], 
        'B': ['mean', 'std'],
        'H': ['mean', 'std'],
        'S': ['mean', 'std'],
        'V': ['mean', 'std'],
        'names': 'count'
    }).round(2)
    
    print("各聚类统计信息:")
    print(cluster_summary)
    
    # 每个聚类的颜色分布
    plt.figure(figsize=(12, 8))
    for cluster_id in range(optimal_k):
        cluster_data = df[df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            plt.scatter(cluster_data['R'], cluster_data['G'], 
                       c=cluster_data[['R', 'G', 'B']].values/255, 
                       label=f'Cluster {cluster_id}', s=50, alpha=0.7)
    
    plt.xlabel('Red')
    plt.ylabel('Green')
    plt.title('Lipstick Color Clusters (RGB Space)')
    plt.legend()
    plt.show()

analyze_clusters(df, features)

def prepare_resnet_data(df, output_file=output_file):
    """
    准备用于ResNet分类训练的数据格式
    """
    def get_cluster_color_name(r_mean, g_mean, b_mean, h_mean):
        # 基于HSV值给聚类命名
        if h_mean < 15 or h_mean > 345:
            return 'reds'
        elif h_mean < 45:
            return 'oranges_corals'
        elif h_mean < 75:
            return 'yellows_beiges'
        elif h_mean < 165:
            return 'pinks'
        elif h_mean < 255:
            return 'purples_berries'
        elif h_mean < 285:
            return 'mauves'
        else:
            return 'deep_reds_burgundies'
    
    # 计算每个聚类的平均颜色特征
    cluster_colors = df.groupby('cluster')[['R', 'G', 'B', 'H', 'S', 'V']].mean()
    
    # 为每个聚类分配有意义的名称
    cluster_names = {}
    color_counts = {}
    
    for cluster_id in range(optimal_k):
        r = cluster_colors.loc[cluster_id, 'R']
        g = cluster_colors.loc[cluster_id, 'G'] 
        b = cluster_colors.loc[cluster_id, 'B']
        h = cluster_colors.loc[cluster_id, 'H']
        s = cluster_colors.loc[cluster_id, 'S']
        v = cluster_colors.loc[cluster_id, 'V']
        
        base_name = get_cluster_color_name(r, g, b, h)
        
        # 根据饱和度和明度添加修饰词
        if s < 0.3:
            shade = 'pale_'
        elif s > 0.7:
            shade = 'vibrant_'
        else:
            shade = ''
            
        if v < 0.4:
            brightness = 'dark_'
        elif v > 0.8:
            brightness = 'bright_'
        else:
            brightness = ''
        
        final_name = f"{brightness}{shade}{base_name}"
        
        # 处理重名
        if final_name in color_counts:
            color_counts[final_name] += 1
            final_name = f"{final_name}_{color_counts[final_name]}"
        else:
            color_counts[final_name] = 1
            
        cluster_names[cluster_id] = final_name
    
    # 添加聚类名称到数据集
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    # 保存处理后的数据
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("聚类分布:")
    print(df['cluster_name'].value_counts())
    
    return df, cluster_names

# 准备ResNet训练数据
df_labeled, cluster_names = prepare_resnet_data(df)
print("\n聚类名称映射:")
for cluster_id, name in cluster_names.items():
    count = len(df_labeled[df_labeled['cluster'] == cluster_id])
    rgb_mean = df_labeled[df_labeled['cluster'] == cluster_id][['R','G','B']].mean().astype(int)
    print(f"Cluster {cluster_id}: {name} ({count}个样本) RGB均值: {tuple(rgb_mean)}")

def visualize_color_clusters(df):
    """
    可视化所有聚类的颜色
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RGB空间分布
    scatter = axes[0,0].scatter(df['R'], df['G'], c=df['cluster'], 
                               cmap='tab10', alpha=0.7)
    axes[0,0].set_xlabel('Red')
    axes[0,0].set_ylabel('Green')
    axes[0,0].set_title('Color Distribution in RGB Space')
    plt.colorbar(scatter, ax=axes[0,0])
    
    # 2. HSV空间分布
    scatter = axes[0,1].scatter(df['H'], df['S'], c=df['cluster'],
                               cmap='tab10', alpha=0.7)
    axes[0,1].set_xlabel('Hue')
    axes[0,1].set_ylabel('Saturation')
    axes[0,1].set_title('Color Distribution in HSV Space')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # 3. 聚类大小分布
    cluster_counts = df['cluster_name'].value_counts()
    axes[1,0].bar(range(len(cluster_counts)), cluster_counts.values)
    axes[1,0].set_xticks(range(len(cluster_counts)))
    axes[1,0].set_xticklabels(cluster_counts.index, rotation=45, ha='right')
    axes[1,0].set_title('Cluster Size Distribution')
    axes[1,0].set_ylabel('Number of Samples')
    
    # 4. 显示每个聚类的代表颜色
    colors_per_row = 5
    for cluster_id in range(optimal_k):
        cluster_data = df[df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_color = cluster_data[['R','G','B']].mean().values / 255
            row = cluster_id // colors_per_row
            col = cluster_id % colors_per_row
            
            if row < 2 and col < 5:  # 确保在子图范围内
                axes[1,1].add_patch(plt.Rectangle((col*0.2, 0.8-row*0.4), 0.18, 0.35, 
                                                color=cluster_color))
                axes[1,1].text(col*0.2 + 0.09, 0.8-row*0.4 - 0.05, 
                              f'C{cluster_id}', ha='center', fontsize=8)
    
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_title('Cluster Representative Colors')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_color_clusters(df_labeled)

# 输出最终结果
print("=" * 50)
print("口红颜色自动聚类完成!")
print("=" * 50)
print(f"数据集大小: {len(df)} 条记录")
print(f"自动确定的聚类数量: {optimal_k}")
print(f"聚类结果已保存到: dataset\lipstick_clusters.csv")

# 显示每个聚类的样本数量
cluster_distribution = df_labeled['cluster_name'].value_counts()
print("\n各颜色类别分布:")
for cluster_type, count in cluster_distribution.items():
    percentage = (count / len(df)) * 100
    print(f"  {cluster_type}: {count}个样本 ({percentage:.1f}%)")

print(f"\n后续ResNet训练建议:")
print(f"  类别数量: {optimal_k}")
print(f"  输入特征: RGB颜色值 (R, G, B) 或 HSV颜色值")
print(f"  输出类别: {list(cluster_names.values())}")

with open(cluster_dict_file, 'w', encoding='utf-8') as dict_label_file:
    for index in range(optimal_k):
        dict_label_file.write(f'{index} {cluster_names[index]}\n')