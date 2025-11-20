import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

input_file = 'datasets/lipstick.csv'
output_file = 'datasets/lipstick_clusters.csv'
cluster_dict_file = 'datasets/dict.csv'

df = pd.read_csv(input_file)
# print(f"the shape of datasets: {df.shape}")
print(df.head())

# Choose attributes: RGB, HSV
features = ['R', 'G', 'B', 'H', 'S', 'V']
X = df[features].values

# Scale Datasets
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("the shape of attributed datasets:", X_scaled.shape)

# Find best K by calculating WCSS and Silhouettes Score 
def find_optimal_k(X, max_k=15):
    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
    
    return wcss, silhouette_scores, k_range

# Plot K curve
def plot_optimal_k(wcss, silhouette_scores, k_range):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # WCSS
    ax1.plot(k_range, wcss, 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Elbow Method for Optimal K')
    ax1.grid(True)
    
    # Silhouettes Score
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal K')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

wcss, silhouette_scores, k_range = find_optimal_k(X_scaled)
plot_optimal_k(wcss, silhouette_scores, k_range)


def auto_select_k(silhouette_scores, k_range, threshold=0.002, display_top_n=5):
    best_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_idx]
    best_score = silhouette_scores[best_idx]
    
    print(f"Best K suggested by Silhouettes: K={best_k}, Silhouettes Score={best_score:.4f}")
    
    sorted_indices = np.argsort(silhouette_scores)[::-1]
    print("\n")
    print(f"Ranking the K in top {display_top_n}:")
    for i, idx in enumerate(sorted_indices[:display_top_n]):
        print(f"{i+1}. K={k_range[idx]}: {silhouette_scores[idx]:.4f}")
    
    return best_k

optimal_k = auto_select_k(silhouette_scores, list(k_range))
print(f"Suggest K: {optimal_k}")

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
    
    print("Cluster information:")
    print(cluster_summary)
    
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

def prepare_train_data(df, output_file=output_file):
    def get_cluster_color_name(r_mean, g_mean, b_mean, h_mean):
        # rename by H (in HSV)
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
    
    # Calculate average color attributes of every Cluster
    cluster_colors = df.groupby('cluster')[['R', 'G', 'B', 'H', 'S', 'V']].mean()
    
    # Rename every Cluster
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
        
        # NO repeated name
        if final_name in color_counts:
            color_counts[final_name] += 1
            final_name = f"{final_name}_{color_counts[final_name]}"
        else:
            color_counts[final_name] = 1
            
        cluster_names[cluster_id] = final_name
    
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("the distribution of cluster:")
    print(df['cluster_name'].value_counts())
    
    return df, cluster_names

df_labeled, cluster_names = prepare_train_data(df)
print("\n")
print("Samples distribution:")
for cluster_id, name in cluster_names.items():
    count = len(df_labeled[df_labeled['cluster'] == cluster_id])
    rgb_mean = df_labeled[df_labeled['cluster'] == cluster_id][['R','G','B']].mean().astype(int)
    print(f"Cluster {cluster_id}: {name} ({count} samples) RGB_mean: {tuple(rgb_mean)}")

# Visualization
def visualize_color_clusters(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RGB
    scatter = axes[0,0].scatter(df['R'], df['G'], c=df['cluster'], 
                               cmap='tab10', alpha=0.7)
    axes[0,0].set_xlabel('Red')
    axes[0,0].set_ylabel('Green')
    axes[0,0].set_title('Color Distribution in RGB Space')
    plt.colorbar(scatter, ax=axes[0,0])
    
    # 2. HSV
    scatter = axes[0,1].scatter(df['H'], df['S'], c=df['cluster'],
                               cmap='tab10', alpha=0.7)
    axes[0,1].set_xlabel('Hue')
    axes[0,1].set_ylabel('Saturation')
    axes[0,1].set_title('Color Distribution in HSV Space')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # 3. Cluster
    cluster_counts = df['cluster_name'].value_counts()
    axes[1,0].bar(range(len(cluster_counts)), cluster_counts.values)
    axes[1,0].set_xticks(range(len(cluster_counts)))
    axes[1,0].set_xticklabels(cluster_counts.index, rotation=45, ha='right')
    axes[1,0].set_title('Cluster Size Distribution')
    axes[1,0].set_ylabel('Number of Samples')
    
    # 4. Display solid color of every cluster
    colors_per_row = 5
    for cluster_id in range(optimal_k):
        cluster_data = df[df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_color = cluster_data[['R','G','B']].mean().values / 255
            row = cluster_id // colors_per_row
            col = cluster_id % colors_per_row
            
            if row < 2 and col < 5:
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

print("=" * 50)
print("Cluster Done!")
print("=" * 50)
print(f"Datasets: {len(df)} samples")
print(f"Cluster value (Best K): {optimal_k}")

cluster_distribution = df_labeled['cluster_name'].value_counts()
print("\n")
print("Distribution of every color:")
for cluster_type, count in cluster_distribution.items():
    percentage = (count / len(df)) * 100
    print(f"{cluster_type}: {count} samples ({percentage:.1f}%)")

print(f"\n")
print(f"For next step: Training")
print(f"num_classes: {optimal_k}")
print(f"Input Attributes: RGB or HSV")
print(f"Dictionary: {list(cluster_names.values())}")

with open(cluster_dict_file, 'w', encoding='utf-8') as dict_label_file:
    for index in range(optimal_k):
        dict_label_file.write(f'{index} {cluster_names[index]}\n')