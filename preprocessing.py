import pandas as pd
import numpy as np
import os
import collections
from sklearn.model_selection import train_test_split

def get_data(path_data, condition_list):
    data_condition = []
    file_list = []
    for condition in condition_list:
        for file in os.listdir(path_data):
            if condition==file.split('_')[1]:
                path_cur = os.path.join(path_data, file)
                data_cur = pd.read_csv(path_cur)
                data_condition.append(data_cur)
                file_list.append(path_cur)
    return data_condition, file_list

def del_columns(path_data, condition):
    data_condition, file_list = get_data(path_data, condition)  
    index_file = 0
    for data_cur in data_condition:
        data_cur = data_cur.loc[:, ~data_cur.columns.str.contains('^Unnamed')]
        data_cur.to_csv(file_list[index_file],index=False)
        index_file += 1

def change_columns(path_data, condition):
    data_condition, file_list = get_data(path_data, condition)  
    index_file = 0
    for data_cur in data_condition:
        user_index = int(file_list[index_file].split('/')[2].split('_')[0])
        data_cur['username'] = user_index
        data_cur.to_csv(file_list[index_file],index=False)
        index_file += 1

class Data_transfer():
    def __init__(self, path_data, condition, encoding_method, num_target=10, seed=42, num_intervl=100):
        print("init dataset")
        self.gesture = condition
        self.path_data = path_data
        self.seed = seed
        self.num_target = num_target
        self.num_intervl = num_intervl
        data_condition, file_list = get_data(path_data, condition)
        self.data_condition = data_condition
        self.file_list = file_list
        self.encoding_method = encoding_method
        self.data_transfer()
        self.save_data()
    def compute_distance(self, x, y):
        return np.sqrt(np.abs(np.square(x)+np.square(y)))
    def save_data(self):
        save_list = ['train', 'val', 'test']
        for save_cur in save_list:
            save_path = f'./data/MTS_{self.encoding_method}_{self.gesture}_{self.seed}_{self.num_intervl}/{save_cur}.txt'
            if save_cur == 'train':
                cur_user = self.train_user
                cur_item = self.train_item
            elif save_cur == 'val':
                cur_user = self.val_user
                cur_item = self.val_item
            elif save_cur == 'test':
                cur_user = self.test_user
                cur_item = self.test_item
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                for i in range(len(cur_user)):
                    items_str = ' '.join(map(str, cur_item[i]))
                    f.write(f'{cur_user[i]} {items_str}\n')

    def data_transfer(self):
        distance_all = []
        W_all = []
        V_all = []
        # get the distance between touch point and target point
        for data_cur in self.data_condition:
            distance_person = []
            for i in range(self.num_target):
                distance_person.append(self.compute_distance(data_cur[f'{i}_touch_x_rad'],data_cur[f'{i}_touch_y_rad']))
            distance_all.append(distance_person)
            W_all.append(data_cur['W'])
            V_all.append(data_cur['V'])
        distance_all_array = np.array(distance_all)
        W_all_array = np.array(W_all)
        V_all_array = np.array(V_all)
        
        # get the unique values of W and V, and the intervals of distance
        self.W_unique = np.unique(W_all_array) # 65,  95, 125, 155
        self.V_unique = np.unique(V_all_array) # 300,  550,  800, 1050
        self.distance_max = np.max(distance_all_array)
        self.distance_min = np.min(distance_all_array)
        self.distance_intervl = (self.distance_max - self.distance_min)/self.num_intervl
        self.m_items = int(len(self.W_unique)*len(self.V_unique)*self.num_intervl)
        # encode the targets as items
        data_train, data_val, data_test = split_train_test(self.data_condition)
        if self.encoding_method == 'base':
            self.train_user, self.train_item = self.encode_base_test(data_train)
            self.val_user, self.val_item = self.encode_base_test(data_val)
            self.test_user, self.test_item = self.encode_base_test(data_test)
        elif self.encoding_method == 'base_wv':
            self.train_user, self.train_item = self.encode_base_wv_test(data_train)
            self.val_user, self.val_item = self.encode_base_wv_test(data_val)
            self.test_user, self.test_item = self.encode_base_wv_test(data_test)
        elif self.encoding_method == 'rank':
            self.train_user, self.train_item = self.encode_rank_test(data_train)
            self.val_user, self.val_item = self.encode_rank_test(data_val)
            self.test_user, self.test_item = self.encode_rank_test(data_test)
    
    def encode_rank_test(self, subset):
        # find the pos of a, b, and c
        user_all = []
        item_all = []
        for count_raw in range(len(subset)):
            item_cur = []
            item_cur.append(subset['W'][count_raw])
            item_cur.append(subset['V'][count_raw])
            item_cur.append(subset['touch_x'][count_raw])
            item_cur.append(subset['touch_y'][count_raw])
            for i in range(self.num_target):
                item_cur.append(subset[f'{i}_target_x'][count_raw])
                item_cur.append(subset[f'{i}_target_y'][count_raw])
                item_cur.append(subset[f'{i}_target_angle'][count_raw])
            user_all.append(subset['username'][count_raw]-1)
            item_all.append(item_cur)
        return user_all, item_all
    
    def encode_base_train(self, subset):
        # find the pos of a, b, and c
        user_all = []
        item_all = []
        for cur_name in range(1,max(subset['username'])+1):
            pos_cur = np.where(subset['username']==cur_name)[0]
            Distance_idx = (self.compute_distance(subset['0_touch_x_rad'][pos_cur],subset['0_touch_y_rad'][pos_cur]) - self.distance_min) // self.distance_intervl
            Distance_idx = np.clip(Distance_idx,0,self.num_intervl-1)
            # final code
            code = Distance_idx
            code_array = np.array(code).astype(np.int16)
            user_all.append(cur_name-1)
            item_all.append(code_array)
        return user_all, item_all
    
    def encode_base_test(self, subset):
        # find the pos of a, b, and c
        user_all = []
        item_all = []
        for count_raw in range(len(subset)):
            item_cur = []
            for i in range(self.num_target):
                Distance_idx = int((self.compute_distance(subset[f'{i}_touch_x_rad'][count_raw],subset[f'{i}_touch_y_rad'][count_raw]) - self.distance_min) // self.distance_intervl)
                Distance_idx = np.clip(Distance_idx,0,self.num_intervl-1)
                # final code
                item_cur.append(Distance_idx)
            user_all.append(subset['username'][count_raw]-1)
            item_all.append(item_cur)
        return user_all, item_all

    def encode_base_wv_train(self, subset):
        # find the pos of a, b, and c
        user_all = []
        item_all = []
        for cur_name in range(1,max(subset['username'])+1):
            pos_cur = np.where(subset['username']==cur_name)[0]
            Distance_idx = (self.compute_distance(subset['0_touch_x_rad'][pos_cur],subset['0_touch_y_rad'][pos_cur]) - self.distance_min) // self.distance_intervl
            Distance_idx = np.clip(Distance_idx,0,self.num_intervl-1)
            W_idx = subset['W'][pos_cur]
            W_idx = np.array([np.where(self.W_unique==value)[0] for value in W_idx]).squeeze()
            V_idx = subset['V'][pos_cur]
            V_idx = np.array([np.where(self.V_unique==value)[0] for value in V_idx]).squeeze()
            # final code
            code = W_idx*len(self.V_unique)*self.num_intervl+V_idx*self.num_intervl+Distance_idx
            code_array = np.array(code).astype(np.int16)
            user_all.append(cur_name-1)
            item_all.append(code_array)
        return user_all, item_all
    
    def encode_base_wv_test(self, subset):
        # find the pos of a, b, and c
        user_all = []
        item_all = []
        for count_raw in range(len(subset)):
            item_cur = []
            for i in range(self.num_target):
                Distance_idx = int((self.compute_distance(subset[f'{i}_touch_x_rad'][count_raw],subset[f'{i}_touch_y_rad'][count_raw]) - self.distance_min) // self.distance_intervl)
                Distance_idx = np.clip(Distance_idx,0,self.num_intervl-1)
                W_idx = subset['W'][count_raw]
                W_idx = np.where(self.W_unique==W_idx)[0][0]
                V_idx = subset['V'][count_raw]
                V_idx = np.where(self.V_unique==V_idx)[0][0]
                code = W_idx*len(self.V_unique)*self.num_intervl+V_idx*self.num_intervl+Distance_idx
                # final code
                item_cur.append(code)
            user_all.append(subset['username'][count_raw]-1)
            item_all.append(item_cur)
        return user_all, item_all

def CoordinateConversion(path_data, condition, num_target):
    data_condition, file_list = get_data(path_data, condition)
    index_file = 0
    for data_cur in data_condition:
        for i in range(num_target):
            data_cur[f'{i}_touch_x_rad'] = data_cur['touch_x'] - data_cur[f'{i}_target_x']
            data_cur[f'{i}_touch_y_rad'] = data_cur['touch_y'] - data_cur[f'{i}_target_y']
            
            direction_rad = np.radians(data_cur[f'{i}_target_angle'])
            tmpX = (data_cur[f'{i}_touch_x_rad'] * np.cos(direction_rad) +
                    data_cur[f'{i}_touch_y_rad'] * np.sin(direction_rad))
            tmpY = (data_cur[f'{i}_touch_y_rad'] * np.cos(direction_rad) -
                    data_cur[f'{i}_touch_x_rad'] * np.sin(direction_rad))
            data_cur[f'{i}_touch_x_rad'] = tmpX
            data_cur[f'{i}_touch_y_rad'] = tmpY
            data_cur.to_csv(file_list[index_file],index=False)
        index_file += 1

def split_train_test(data_condition, test_ratio=0.3, val_ratio=0.1):
    """
    Split the data into training and test sets
    """
    W_values = list(collections.Counter(data_condition[0]['W']).keys())
    V_values = list(collections.Counter(data_condition[0]['V']).keys())

    data_train, data_val, data_test = [], [], []

    for data_cur in data_condition:
        for w in W_values:
            for v in V_values:
                # Filter data by current W and V combination
                data_subset = data_cur[(data_cur['W'] == w) & (data_cur['V'] == v)]
                
                # Split into train, validation, and test sets
                train_data, temp_data = train_test_split(data_subset, test_size=test_ratio + val_ratio)
                val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio))
                
                data_train.append(train_data)
                data_val.append(val_data)
                data_test.append(test_data)

    data_train = pd.concat(data_train, ignore_index=True)
    data_val = pd.concat(data_val, ignore_index=True)
    data_test = pd.concat(data_test, ignore_index=True)
    
    return data_train, data_val, data_test

# for file analysis

def CoordinateConversion_file(path_data, num_target):
    data_cur = pd.read_csv(path_data)
    index_file = 0
    
    for i in range(num_target):
        data_cur[f'{i}_touch_x_rad'] = data_cur['touch_x'] - data_cur[f'{i}_target_x']
        data_cur[f'{i}_touch_y_rad'] = data_cur['touch_y'] - data_cur[f'{i}_target_y']
        direction_rad = np.radians(data_cur[f'{i}_target_angle'])
        tmpX = (data_cur[f'{i}_touch_x_rad'] * np.cos(direction_rad) +
                data_cur[f'{i}_touch_y_rad'] * np.sin(direction_rad))
        tmpY = (data_cur[f'{i}_touch_y_rad'] * np.cos(direction_rad) -
                data_cur[f'{i}_touch_x_rad'] * np.sin(direction_rad))
        data_cur[f'{i}_touch_x_rad'] = tmpX
        data_cur[f'{i}_touch_y_rad'] = tmpY
        data_cur.to_csv(path_data,index=False)
    index_file += 1

if __name__ == '__main__':
    acc_file = pd.read_csv('./dataset/acc_total.csv')
    endpoints_file = pd.read_csv('./dataset/endpoints_total.csv')
    rmsa_file = pd.read_csv('./dataset/rmsa_total.csv',header=None)
    vib_file = pd.read_csv('./dataset/vib_total.csv',encoding='gbk')
    import ipdb;ipdb.set_trace()
    rows_to_remove = np.where(rmsa_file>5)[0]
    acc_file.drop(rows_to_remove, inplace=True)
    endpoints_file.drop(rows_to_remove, inplace=True)
    rmsa_file.drop(rows_to_remove, inplace=True)
    vib_file.drop(rows_to_remove, inplace=True)

    acc_file.to_csv('./dataset/acc_total.csv', index=False)
    endpoints_file.to_csv('./dataset/endpoints_total.csv', index=False)
    rmsa_file.to_csv('./dataset/rmsa_total.csv', index=False)
    vib_file.to_csv('./dataset/vib_total.csv', index=False)


    #CoordinateConversion_file(f'./dataset/endpoints_total.csv', 15)
#split_train_test('./data_source', ['1'])
# data_transfer = Data_transfer('./data_source', '1', 'rank', 10, 0, num_intervl=50)
# del_columns('./data_source', ['2'])
# change_columns('./data_source', ['2'])