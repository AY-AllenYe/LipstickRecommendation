import csv

def csv_to_dict(file_path):
    data = []
    my_dict = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label = line.strip().split()[0]
            name = line.strip().split()[1]
            
            data.append((label, name))
    my_dict = dict(data)
    return my_dict

# cluster_dict_file = 'datasets/dict.csv'
# my_dict = csv_to_dict_list(cluster_dict_file)
# print(my_dict)