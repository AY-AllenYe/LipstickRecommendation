import pandas as pd
import random

def recommendation(cluster_file, pred_cluster_label, rec_samples, random_seed=42):
    
    df = pd.read_csv(cluster_file)
    cluster_data = df[df['cluster'] == pred_cluster_label]
    
    if len(cluster_data) < rec_samples:
        print(f"Sorry the repository don't have so much. Here is all we save, total {len(cluster_data)} items.")
        print('\n')
    
    actual_samples = min(rec_samples, len(cluster_data))
    # sampled_data = cluster_data.sample(n=actual_samples, random_state=random_seed)
    
    
    
    cluster_data_reset = cluster_data.reset_index(drop=True)
    available_indices = list(range(len(cluster_data_reset)))
    selected_indices = []
    for _ in range(actual_samples):
        if not available_indices:
            break
            
        random_index = random.choice(available_indices)
        selected_indices.append(random_index)
        available_indices.remove(random_index)

    sampled_data_list = []
    for idx in selected_indices:
        sampled_data_list.append(cluster_data_reset.iloc[idx])
    
    for index in range(rec_samples):
        if not pd.isna(sampled_data_list[index]['names']) :
            print(f"{sampled_data_list[index]['brands']} - {sampled_data_list[index]['series']} - {sampled_data_list[index]['names']}, {sampled_data_list[index]['id']}")
        else:
            print(f"{sampled_data_list[index]['brands']} - {sampled_data_list[index]['series']} - (Unnamed), {sampled_data_list[index]['id']}")
    
    # return sampled_data_list

# cluster_file = 'datasets/lipstick_clusters.csv'
# pred_label = 1
# mode_rec_num = 10
# sampled_data = recommendation(cluster_file, pred_label, mode_rec_num)
# for index in range(mode_rec_num):
#     if not pd.isna(sampled_data[index]['names']) :
#         print(f"{sampled_data[index]['brands']} - {sampled_data[index]['series']} - {sampled_data[index]['names']}, {sampled_data[index]['id']}")
#     else:
#         print(f"{sampled_data[index]['brands']} - {sampled_data[index]['series']} - (Unnamed), {sampled_data[index]['id']}")