import utils
import torch
import numpy as np
import time
from os.path import join
import os
from preprocessing import split_train_test
from parse import parse_args
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset, DataLoader
from torch.distributions import MultivariateNormal, MixtureSameFamily
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn as nn
import torch.nn.functional as F

args = parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda")
    num_workers = 2
else:
    device = torch.device("cpu")
    num_workers = 0

args.num_target = 15
args.dim_user = 5
epoch_early_stop = 100
early_stop_patience = 10

kwargs = {'num_workers': num_workers, 'pin_memory': True}
condition_name = ["W", "V", "E"]
vib_name = ["vel_x_1", "vel_y_1", "vel_z_1", "ang_x_1", "ang_y_1", "ang_z_1", "disp_x_1", "disp_y_1", "disp_z_1", "freq_x_1", "freq_y_1", "freq_z_1", \
            "vel_x_2", "vel_y_2", "vel_z_2", "ang_x_2", "ang_y_2", "ang_z_2", "disp_x_2", "disp_y_2", "disp_z_2", "freq_x_2", "freq_y_2", "freq_z_2", \
            "vel_x_3", "vel_y_3", "vel_z_3", "ang_x_3", "ang_y_3", "ang_z_3", "disp_x_3", "disp_y_3", "disp_z_3", "freq_x_3", "freq_y_3", "freq_z_3", \
            "vel_x_4", "vel_y_4", "vel_z_4", "ang_x_4", "ang_y_4", "ang_z_4", "disp_x_4", "disp_y_4", "disp_z_4", "freq_x_4", "freq_y_4", "freq_z_4", \
            "vel_x_5", "vel_y_5", "vel_z_5", "ang_x_5", "ang_y_5", "ang_z_5", "disp_x_5", "disp_y_5", "disp_z_5", "freq_x_5", "freq_y_5", "freq_z_5", \
            "vel_x_6", "vel_y_6", "vel_z_6", "ang_x_6", "ang_y_6", "ang_z_6", "disp_x_6", "disp_y_6", "disp_z_6", "freq_x_6", "freq_y_6", "freq_z_6", \
            "vel_x_7", "vel_y_7", "vel_z_7", "ang_x_7", "ang_y_7", "ang_z_7", "disp_x_7", "disp_y_7", "disp_z_7", "freq_x_7", "freq_y_7", "freq_z_7", \
            "vel_x_8", "vel_y_8", "vel_z_8", "ang_x_8", "ang_y_8", "ang_z_8", "disp_x_8", "disp_y_8", "disp_z_8", "freq_x_8", "freq_y_8", "freq_z_8"]
acc_name = ["acc_x_1", "acc_y_1", "acc_z_1", "acc_x_2", "acc_y_2", "acc_z_2", "acc_x_3", "acc_y_3", "acc_z_3",\
             "acc_x_4", "acc_y_4", "acc_z_4", "acc_x_5", "acc_y_5", "acc_z_5", "acc_x_6", "acc_y_6", "acc_z_6", 
             "acc_x_7", "acc_y_7", "acc_z_7", "acc_x_8", "acc_y_8", "acc_z_8"]

split_cluster = 0.577
split_median = 0.4642
split_mean = 0.5508

def compute_hit(predicts_all, data_cur):
    precision_list = []
    for i in range(1,predicts_all.shape[1]+1):
        cur_hit = predicts_all[:,0:i]
        num_hit = torch.where(cur_hit==0)[0].shape[0]
        precision_list.append(num_hit/data_cur.shape[0])
    precision = np.array(precision_list)
    return precision

class BuildDataset(Dataset):
    def __init__(self, data_all, num_target=15):
        condition_all = []
        points_all = []
        label_all = []
        rmsa_all = []
        vib_all = []
        acc_all = []
        gestures = data_all['gesture'].unique()
        genders = data_all['usergender'].unique()
        
        for index, data in data_all.iterrows():
            gesture_one_hot = np.zeros(len(gestures))
            gesture_one_hot[np.where(gestures == data['gesture'])[0][0]] = 1
            
            userage_normalized = (data['userage'] - data_all['userage'].min()) / (data_all['userage'].max() - data_all['userage'].min())
            
            gender_one_hot = np.zeros(len(genders))
            gender_one_hot[np.where(genders == data['usergender'])[0][0]] = 1
            
            condition = np.concatenate(([data['W'], data['V'], data['rmsa']], gesture_one_hot, [userage_normalized], gender_one_hot))
            condition_all.append(condition)
            
            points = []
            rmsa_all.append(data.iloc[89])
            acc_all.append(data[90:114])
            vib_all.append(data[114:])
            for i in range(num_target):
                cur_pos = np.array([data[f'{i}_touch_x_rad'], data[f'{i}_touch_y_rad']])
                points.append(cur_pos)
            points = np.array(points).transpose(0, 1)
            label_original = [0 for i in range(num_target)]
            label_original[0] = 1
            label_original = np.array(label_original)
            random_index = np.random.permutation(num_target)
            label_random = label_original[random_index]
            points_random = points[random_index]
            points_all.append(points_random)
            label_all.append(label_random)
        
        self.condition = np.array(condition_all).squeeze()
        self.points = np.array(points_all)
        self.label = np.array(label_all)
        self.rmsa = np.array(rmsa_all)
        self.vib = np.array(vib_all)
        self.acc = np.array(acc_all)

    def __getitem__(self, index):
        acc = self.acc[index].reshape(8, 3)
        vib = self.vib[index].reshape(8, 12)
        return self.condition[index], self.points[index],  [self.rmsa[index], vib, acc], self.label[index]
    
    def __len__(self):
        return len(self.label)

def split_train_test_few(data_condition, num_shot=1, seed=0, test_ratio=0.4, val_ratio=0.1):
    W_values = list(collections.Counter(data_condition['W']).keys())
    V_values = list(collections.Counter(data_condition['V']).keys())

    data_train, data_val, data_test = [], [], []
    num_user = len(data_condition['username'].unique())
    for user in range(1,num_user+1):
        user_data_condition = data_condition[data_condition['username'] == user]
        for w in W_values:
            for v in V_values:
                data_subset = user_data_condition[(user_data_condition['W'] == w) & (user_data_condition['V'] == v)]
                
                train_data_candidate, temp_data = train_test_split(data_subset, test_size=test_ratio + val_ratio)
                val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio))
                train_data = train_data_candidate.sample(n=num_shot, random_state=seed)
                
                data_train.append(train_data)
                data_val.append(val_data)
                data_test.append(test_data)

    data_train = pd.concat(data_train, ignore_index=True)
    data_val = pd.concat(data_val, ignore_index=True)
    data_test = pd.concat(data_test, ignore_index=True)
    
    return data_train, data_val, data_test

def split_train_test(data_condition, test_ratio=0.3, val_ratio=0.1):
    W_values = list(collections.Counter(data_condition['W']).keys())
    V_values = list(collections.Counter(data_condition['V']).keys())

    data_train, data_val, data_test = [], [], []

    for w in W_values:
        for v in V_values:
            data_subset = data_condition[(data_condition['W'] == w) & (data_condition['V'] == v)]
            
            train_data, temp_data = train_test_split(data_subset, test_size=test_ratio + val_ratio)
            val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio))
            
            data_train.append(train_data)
            data_val.append(val_data)
            data_test.append(test_data)

    data_train = pd.concat(data_train, ignore_index=True)
    data_val = pd.concat(data_val, ignore_index=True)
    data_test = pd.concat(data_test, ignore_index=True)
    
    return data_train, data_val, data_test

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, hn = self.gru(x)
        out = self.dropout(out)
        return out, hn

class FusionModel(nn.Module):
    def __init__(self, paras, args):
        super(FusionModel, self).__init__()
        self.args = args
        self.num_target = args.num_target
        self.num_expert = self.args.num_expert
        # Base parameters with proper initialization
        #self.register_buffer('base_paras', torch.tensor(paras.to_numpy()).float())
        self.base_paras = nn.Parameter(torch.randn(self.num_expert, args.dim_expert) * 0.1, requires_grad=False)
        self.adapted_paras = nn.Parameter(self.base_paras.clone(), requires_grad=True)
        self.weights_expert = nn.Parameter(torch.tensor([0.,1.,0.]))
        # Improved parameter adaptation networks
        context_dim = args.dim_user + args.hidden_dim * 4

        # User encoder
        self.encoder_user = nn.Linear(args.dim_user, args.hidden_dim)
        # Touch feature encoder
        self.encoder_touch = nn.Linear(args.dim_touch, args.hidden_dim)
        # Environment encoders with attention
        self.encoder_vib = GRUEncoder(12, args.hidden_dim, 2)
        self.encoder_acc = GRUEncoder(3, args.hidden_dim, 2)
        
        # Attention for environment features
        self.vib_attention = nn.MultiheadAttention(args.hidden_dim * 2, 8, dropout=0.1)
        self.acc_attention = nn.MultiheadAttention(args.hidden_dim * 2, 8, dropout=0.1)
        
        # Improved fusion network with diversity regularization
        fusion_input_dim = args.hidden_dim * 6
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, args.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(args.hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dim, args.num_expert)
        )
        # Adaptive parameters
        self.adapt_parameters = nn.Parameter(torch.randn(args.num_expert, self.args.dim_expert), requires_grad=True)

        # Temperature for fusion weights - initialize higher to encourage diversity
        self.temperature = nn.Parameter(torch.tensor(self.args.temperature), requires_grad=True)
        
        # Expert diversity regularization parameters
        self.diversity_weight = self.args.w_diverse
        
        # Balance penalty to prevent single expert dominance
        self.balance_penalty = self.args.w_balance
        
        self.use_env = args.use_env

    def extract_user_features(self, feat_user):
        """Extract user features"""        
        user_out = self.encoder_user(feat_user)
        
        return user_out
    
    def extract_touch_features(self, feat_touch):
        """Extract touch features"""        
        touch_out = self.encoder_touch(feat_touch)

        return touch_out

    def forward_expert(self, condition, touch_xy, env=None):
        # condition (W, V, E): (bs, 3)
        # touch_xy (touch_x, touch_y): (bs, self.num_target , 2)
        # env: env input
        # self.paras: (self.num_expert, self.dim_expert)
        condition = condition.float()
        size_batch = touch_xy.shape[0]
        fusion_weight = self.weights_expert.unsqueeze(0).repeat(size_batch, 1)
        self.fusion_weight = fusion_weight
        # Only use the first expert (index 0)
        param_cur = self.base_paras[self.weights_expert==1]  # (1, self.dim_expert)
        param_cur = param_cur.unsqueeze(0).expand(size_batch, -1, -1)  # (bs, 1, self.dim_expert)
        MU, SIGMA = self.calc_mu_sigma(param_cur, condition)
        
        # MU: (bs, 1, 2), SIGMA: (bs, 1, 2, 2)
        # Remove the expert dimension since we only use one expert
        MU = MU.squeeze(1)  # (bs, 2)
        SIGMA = SIGMA.squeeze(1)  # (bs, 2, 2)
        
        prob_all = []
        for cur_target in range(self.num_target):
            target_cur = touch_xy[:, cur_target, :]  # (bs, 2)
            # Directly use MultivariateNormal without mixture
            mvn = torch.distributions.MultivariateNormal(
                loc=MU,
                covariance_matrix=SIGMA
            )
            log_probs_tensor = mvn.log_prob(target_cur)
            prob_all.append(log_probs_tensor.unsqueeze(1))
        prob_all = torch.stack(prob_all, dim=0).squeeze().transpose(0, 1)
        return prob_all   

    def extract_env_features(self, vib, acc):
        """Extract environment features with attention mechanism"""
        bs = vib.shape[0]
        
        # GRU encoding
        vib_out, vib_hn = self.encoder_vib(vib)
        acc_out, acc_hn = self.encoder_acc(acc)
        
        # Attention mechanism
        vib_out = vib_out.permute(1, 0, 2)
        acc_out = acc_out.permute(1, 0, 2)
        
        vib_attn, _ = self.vib_attention(vib_out, vib_out, vib_out)
        acc_attn, _ = self.acc_attention(acc_out, acc_out, acc_out)
        
        # Global average pooling
        feat_vib = vib_attn.mean(dim=0)
        feat_acc = acc_attn.mean(dim=0)
        
        return torch.cat([feat_vib, feat_acc], dim=1)
    
    def calc_mu_sigma(self, para, x):
        """Calculate mu and sigma with numerical stability"""
        bs = x.shape[0]
        x1, x2, x3 = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), x[:, 2].unsqueeze(1)

        mu_x = para[:, :, 0] + para[:, :, 1] * x2 + para[:, :, 2] * x1 + para[:, :, 3] * x3
        mu_y = torch.zeros(bs, para.shape[1], device=para.device)

        # Add numerical stability
        eps = 1e-6
        sigma_x = torch.sqrt(torch.abs(para[:, :, 4]) + torch.abs(para[:, :, 5]) * x2**2 + 
                           torch.abs(para[:, :, 6]) * x1**2 + torch.abs(para[:, :, 7]) * (x2 / (x1 + eps)) + 
                           torch.abs(para[:, :, 8]) * x3**2 + eps)
        sigma_y = torch.sqrt(torch.abs(para[:, :, 9]) + torch.abs(para[:, :, 10]) * x2**2 + 
                           torch.abs(para[:, :, 11]) * x1**2 + torch.abs(para[:, :, 12]) * x3**2 + eps)

        MU = torch.stack((mu_x, mu_y), dim=2)
        SIGMA = torch.zeros(bs, para.shape[1], 2, 2, device=para.device)
        SIGMA[:, :, 0, 0] = sigma_x**2
        SIGMA[:, :, 1, 1] = sigma_y**2

        return MU, SIGMA
    
    def calculate_expert_diversity_loss(self):
        """Calculates a loss to encourage experts to be different."""
        if self.num_expert <= 1:
            return 0.0
            
        normed_paras = F.normalize(self.adapted_paras, p=2, dim=1)
        cosine_sim = torch.mm(normed_paras, normed_paras.t())
        mask = torch.triu(torch.ones_like(cosine_sim), diagonal=1).bool()
        diversity_loss = cosine_sim[mask].mean()
        return diversity_loss

    def calculate_diversity_loss(self, fusion_weight):
        """Calculate diversity loss to encourage balanced expert usage"""
        # Entropy regularization to encourage diversity
        entropy_loss = -torch.mean(torch.sum(fusion_weight * torch.log(fusion_weight + 1e-8), dim=1))
        
        # Balance penalty to prevent single expert dominance
        avg_weights = torch.mean(fusion_weight, dim=0)
        uniform_dist = torch.ones_like(avg_weights) / self.num_expert
        balance_loss = F.mse_loss(avg_weights, uniform_dist)
        
        return entropy_loss + self.balance_penalty * balance_loss
        
    def forward(self, condition, touch_xy, env=None):
        condition = condition.float()
        condition_touch = condition[:, :3]
        condition_user = condition[:, 3:]
        size_batch = touch_xy.shape[0]
        
        # Extract environment features
        if env is not None and self.use_env:
            [_, vib, acc] = env
            vib = vib.float()
            acc = acc.float()
            touch_xy = touch_xy.float()
            condition_user = condition_user.float()
            env_features = self.extract_env_features(vib, acc)
            user_features = self.extract_user_features(condition_user)
            touch_features = self.extract_touch_features(condition_touch)
        else:
            env_features = torch.zeros(size_batch, self.args.hidden_dim*4).to(device)
        adapted_paras = self.adapted_paras.unsqueeze(0).expand(size_batch, -1, -1)
        # Calculate Gaussian parameters
        MU, SIGMA = self.calc_mu_sigma(adapted_paras, condition_touch)
        
        # Context-aware fusion weights with diversity regularization
        fusion_input = torch.cat([env_features, user_features, touch_features], dim=1)
        fusion_logits = self.fusion_network(fusion_input)
        # Min-max normalization
        fusion_weight = F.softmax(fusion_logits / self.temperature, dim=1)
        self.fusion_weight = fusion_weight
        
        # Calculate diversity loss
        #self.diversity_loss = self.calculate_diversity_loss(fusion_weight)
        self.diversity_loss = self.calculate_expert_diversity_loss()
        
        # Calculate probabilities
        prob_all = []
        for cur_target in range(self.num_target):
            target_cur = touch_xy[:, cur_target, :]
            
            mix = torch.distributions.Categorical(probs=fusion_weight)
            comp = torch.distributions.MultivariateNormal(
                loc=MU,
                covariance_matrix=SIGMA
            )
            
            gmm = MixtureSameFamily(mix, comp)
            log_probs_tensor = gmm.log_prob(target_cur)
            prob_all.append(log_probs_tensor.unsqueeze(1))
        
        prob_all = torch.stack(prob_all, dim=0).squeeze().transpose(0, 1)
        return prob_all

# Main training loop modifications
if not os.path.exists('./random_number.npy'):
    seed_array = np.random.randint(0, 1e3, 5)
    np.save('./random_number.npy', seed_array)
seed_array = np.load('./random_number.npy')
print("seed array:", seed_array)
result_file = f"{args.result_path}"
if not os.path.exists(result_file):
    os.mkdir(args.result_path)
file_save = open(result_file+'/result.txt','w')


para = pd.read_csv('./dataset_2d/parameters.csv')
para = para.drop(para.columns[0], axis=1)
endpoints = pd.read_csv('./dataset_2d/endpoints_total.csv')
rmsa = pd.read_csv('./dataset_2d/rmsa_total.csv')
acc = pd.read_csv('./dataset_2d/acc_total.csv')
vib = pd.read_csv('./dataset_2d/vib_total.csv')

data_original = pd.concat([endpoints, rmsa, acc, vib], axis=1)

experts = para.to_numpy()
args.num_expert = 3
args.dim_expert = para.shape[1]
args.hidden_dim = 64
args.dim_touch = 3
precision_all = []
cluster_mean_all = []
rmsa_mean_all = []
rmsa_median_all = []
error_all = []

for seed in seed_array:
    utils.set_seed(seed)
    best_model_path = result_file+f'/best_model_seed_{seed}.pth'
    patience_counter = 0
    best_val_loss = float('inf')
    print(">>SEED:", seed)
    
    if args.dataset_split == 'few_shot':
        data_train, data_val, data_test = split_train_test_few(data_original, num_shot=args.num_shot,seed=seed)
        data_train = BuildDataset(data_train, args.num_target)
        data_val = BuildDataset(data_val, args.num_target)
        data_test = BuildDataset(data_test, args.num_target)

        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        dataloader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=True, **kwargs)
        dataloader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        data_train, data_val, data_test = split_train_test(data_original)
        data_train = BuildDataset(data_train, args.num_target)
        data_val = BuildDataset(data_val, args.num_target)
        data_test = BuildDataset(data_test, args.num_target)

        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        dataloader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=True, **kwargs)
        dataloader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    model = FusionModel(para, args)
    model = model.to(device)
    
    # Improved optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_precision = 0.0
    no_improve_epoch = 0
    use_test = False

    for epoch in range(args.epochs):
        precision_train_top1 = 0
        precision_train_top2 = 0
        precision_train_top3 = 0
        loss_all_train = 0
        model.train()
        
        for i, (condition, touch_xy, env, label) in enumerate(dataloader_train):
            size_batch = condition.shape[0]
            condition = condition.to(device)
            touch_xy = touch_xy.to(device)
            label = label.to(device)
            [rmsa,vib,acc] = env
            rmsa = rmsa.to(device)
            vib = vib.to(device)
            acc = acc.to(device)
            
            prob = model(condition, touch_xy, [rmsa,vib,acc])
            
            # Ranking Loss with diversity regularization
            ranking_loss = nn.MarginRankingLoss(margin=1.0)
            positive_scores = prob.gather(1, label.argmax(dim=1).unsqueeze(1))
            negative_mask = (1 - label).bool()
            negative_scores = prob[negative_mask].view(prob.shape[0], -1)
            max_negative_scores, _ = negative_scores.max(dim=1, keepdim=True)

            target = torch.ones(prob.shape[0], 1).to(device)
            ranking_loss_val = ranking_loss(positive_scores, max_negative_scores, target)
            
            # Add diversity loss to encourage balanced expert usage
            diversity_loss_val = model.diversity_loss
            
            # Total loss with diversity regularization
            loss = ranking_loss_val + model.diversity_weight * diversity_loss_val
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()
            loss_all_train += loss.item()
            
            # Evaluation
            topk = 3
            _, topk_pred = prob.topk(topk, dim=1)
            label_true = label.argmax(dim=1).unsqueeze(1).expand(-1, topk)
            precision_train_top1 += (topk_pred[:, 0] == label_true[:, 0]).sum().item()
            precision_train_top2 += (topk_pred[:, :2] == label_true[:,:2]).any(dim=1).sum().item()
            precision_train_top3 += (topk_pred[:, :3] == label_true).any(dim=1).sum().item()
            
        precision_train_top1 /= len(data_train)
        precision_train_top2 /= len(data_train)
        precision_train_top3 /= len(data_train)
        loss_all_train /= len(dataloader_train)
        print(f"train epoch:{epoch}, loss:{loss_all_train}, precision@1:{precision_train_top1:4f}, precision@2:{precision_train_top2:4f}, precision@3:{precision_train_top3:4f}")
        scheduler.step()

        # Validation
        precision_val_top1 = 0
        precision_val_top2 = 0
        precision_val_top3 = 0
        loss_all_val = 0
        model.eval()
        for i, (condition, touch_xy, env, label) in enumerate(dataloader_val):
            with torch.no_grad():
                condition = condition.to(device)
                touch_xy = touch_xy.to(device)
                label = label.to(device)
                [rmsa,vib,acc] = env
                rmsa = rmsa.to(device)
                vib = vib.to(device)
                acc = acc.to(device)
                prob = model(condition, touch_xy, [rmsa,vib,acc])
                ranking_loss = nn.MarginRankingLoss(margin=1.0)
                positive_scores = prob.gather(1, label.argmax(dim=1).unsqueeze(1))
                negative_mask = (1 - label).bool()
                negative_scores = prob[negative_mask].view(prob.shape[0], -1)
                max_negative_scores, _ = negative_scores.max(dim=1, keepdim=True)

                target = torch.ones(prob.shape[0], 1).to(device)
                ranking_loss_val = ranking_loss(positive_scores, max_negative_scores, target)
                
                # Add diversity loss to encourage balanced expert usage
                diversity_loss_val = model.diversity_loss
                
                # Total loss with diversity regularization
                loss = ranking_loss_val + model.diversity_weight * diversity_loss_val

                _, topk_pred = prob.topk(3, dim=1)
                label_true = label.argmax(dim=1).unsqueeze(1).expand(-1, 3)
                precision_val_top1 += (topk_pred[:, 0] == label_true[:, 0]).sum().item()
                precision_val_top2 += (topk_pred[:, :2] == label_true[:,:2]).any(dim=1).sum().item()
                precision_val_top3 += (topk_pred[:, :3] == label_true).any(dim=1).sum().item()
                loss_all_val += loss.item()
        precision_val_top1 /= len(data_val)
        precision_val_top2 /= len(data_val)
        precision_val_top3 /= len(data_val)
        loss_all_val /= len(dataloader_val)
        print(f"val epoch:{epoch}, loss:{loss_all_val:4f}, precision@1:{precision_val_top1:4f}, precision@2:{precision_val_top2:4f}, precision@3:{precision_val_top3:4f}")

        if loss_all_val < best_val_loss:
            best_val_loss = loss_all_val
            patience_counter = 0
            #print(f"  -> New best model found! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{early_stop_patience}")
        if patience_counter >= early_stop_patience or epoch == args.epochs - 1:
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
            if epoch == args.epochs - 1:
                print("Reached maximum epochs without early stopping.")

            precision_test_top1 = 0
            precision_test_top2 = 0
            precision_test_top3 = 0
            precision_cluster_1 = 0
            precision_cluster_2 = 0
            precision_median_1 = 0
            precision_median_2 = 0
            precision_mean_1 = 0
            precision_mean_2 = 0
            num_cluster_1 = 0
            num_cluster_2 = 0
            num_median_1 = 0
            num_median_2 = 0
            num_mean_1 = 0
            num_mean_2 = 0
            loss_all_test = 0
            error_condition = []
            error_touch = []
            error_acc = []
            error_vib = []
            error_label = []
            error_prob = []
            error_weight = []
            for i, (condition, touch_xy, env, label) in enumerate(dataloader_test):
                with torch.no_grad():
                    condition = condition.to(device)
                    touch_xy = touch_xy.to(device)
                    label = label.to(device)
                    [rmsa,vib,acc] = env
                    rmsa = rmsa.to(device)
                    vib = vib.to(device)
                    acc = acc.to(device)
                    prob = model(condition, touch_xy, [rmsa,vib,acc])
                    _, topk_pred = prob.topk(3, dim=1)
                    label_true = label.argmax(dim=1).unsqueeze(1).expand(-1, 3)
                    precision_test_top1 += (topk_pred[:, 0] == label_true[:, 0]).sum().item()
                    precision_test_top2 += (topk_pred[:, :2] == label_true[:,:2]).any(dim=1).sum().item()
                    precision_test_top3 += (topk_pred[:, :3] == label_true).any(dim=1).sum().item()
                    #split_cluster
                    ind_cluster_1 = torch.where(rmsa < split_cluster)[0]
                    ind_cluster_2 = torch.where(rmsa >= split_cluster)[0]
                    num_cluster_1 += ind_cluster_1.shape[0]
                    num_cluster_2 += ind_cluster_2.shape[0]
                    precision_cluster_1 += (topk_pred[ind_cluster_1, 0] == label_true[ind_cluster_1, 0]).sum().item()
                    precision_cluster_2 += (topk_pred[ind_cluster_2, 0] == label_true[ind_cluster_2, 0]).sum().item()
                    #split_median
                    ind_median_1 = torch.where(rmsa < split_median)[0]
                    ind_median_2 = torch.where(rmsa >= split_median)[0]
                    num_median_1 += ind_median_1.shape[0]
                    num_median_2 += ind_median_2.shape[0]
                    precision_median_1 += (topk_pred[ind_median_1, 0] == label_true[ind_median_1, 0]).sum().item()
                    precision_median_2 += (topk_pred[ind_median_2, 0] == label_true[ind_median_2, 0]).sum().item()
                    #split_mean
                    ind_mean_1 = torch.where(rmsa < split_mean)[0]
                    ind_mean_2 = torch.where(rmsa >= split_mean)[0]
                    num_mean_1 += ind_mean_1.shape[0]
                    num_mean_2 += ind_mean_2.shape[0]
                    precision_mean_1 += (topk_pred[ind_mean_1, 0] == label_true[ind_mean_1, 0]).sum().item()
                    precision_mean_2 += (topk_pred[ind_mean_2, 0] == label_true[ind_mean_2, 0]).sum().item()
                    
                    # save error data
                    error_ind = (topk_pred[:, 0] != label_true[:, 0])
                    error_condition.append(condition[error_ind].cpu())
                    error_touch.append(touch_xy[error_ind].cpu())
                    error_acc.append(acc[error_ind].cpu())
                    error_vib.append(vib[error_ind].cpu())
                    error_label.append(label[error_ind].cpu())
                    error_prob.append(prob[error_ind].cpu())
                    error_weight.append(model.fusion_weight[error_ind].cpu())
            
            precision_test_top1 /= len(data_test)
            precision_test_top2 /= len(data_test)
            precision_test_top3 /= len(data_test)
            precision_cluster_1 /= num_cluster_1
            precision_cluster_2 /= num_cluster_2
            precision_median_1 /= num_median_1
            precision_median_2 /= num_median_2
            precision_mean_1 /= num_mean_1
            precision_mean_2 /= num_mean_2
            
            error_condition = torch.cat(error_condition, dim=0)
            error_touch = torch.cat(error_touch, dim=0)
            error_acc = torch.cat(error_acc, dim=0)
            error_vib = torch.cat(error_vib, dim=0)
            error_label = torch.cat(error_label, dim=0)
            error_prob = torch.cat(error_prob, dim=0)
            error_weight = torch.cat(error_weight, dim=0)
            
            error_acc = error_acc.reshape(error_acc.shape[0], -1)
            error_vib = error_vib.reshape(error_vib.shape[0], -1)
            print(f"test epoch:{epoch}, precision@1:{precision_test_top1:4f}, precision@2:{precision_test_top2:4f}, precision@3:{precision_test_top3:4f}")
            print(f"precision_cluster_1:{precision_cluster_1:4f}, precision_cluster_2:{precision_cluster_2:4f}")
            print(f"precision_median_1:{precision_median_1:4f}, precision_median_2:{precision_median_2:4f}")
            print(f"precision_mean_1:{precision_mean_1:4f}, precision_mean_2:{precision_mean_2:4f}")
            
            precision_all.append([precision_test_top1, precision_test_top2, precision_test_top3])
            cluster_mean_all.append([precision_cluster_1, precision_cluster_2])
            rmsa_mean_all.append([precision_median_1, precision_median_2])  
            rmsa_median_all.append([precision_mean_1, precision_mean_2])
            gt_touch = error_touch[range(error_touch.shape[0]),error_label.argmax(dim=1),:].cpu()
            top_touch = error_touch[range(error_touch.shape[0]),error_prob.argmax(dim=1),:].cpu()

            data = {
                'gt_x': np.round(gt_touch[:,0].numpy(), 4).tolist(),
                'gt_y': np.round(gt_touch[:,1].numpy(), 4).tolist(),
                'top_x': np.round(top_touch[:,0].numpy(), 4).tolist(),
                'top_y': np.round(top_touch[:,1].numpy(), 4).tolist(),
                'error_weight': np.round(error_weight.numpy(), 4).tolist()
            }
            for i, name in enumerate(condition_name):
                data[name] = error_condition[:, i].numpy().tolist()
            for i, name in enumerate(acc_name):
                data[name] = error_acc[:, i].numpy().tolist()
            for i, name in enumerate(vib_name):
                data[name] = error_vib[:, i].numpy().tolist()
            
            df = pd.DataFrame(data)
            df.to_csv(result_file+f'/error_data_{seed}.csv', index=False)
            break
mean_precision = np.mean(precision_all, axis=0)
std_precision = np.std(precision_all, axis=0)
mean_cluster = np.mean(cluster_mean_all, axis=0)
std_cluster = np.std(cluster_mean_all, axis=0)
mean_rmsa = np.mean(rmsa_mean_all, axis=0)
std_rmsa = np.std(rmsa_mean_all, axis=0)
mean_rmsa_median = np.mean(rmsa_median_all, axis=0)
std_rmsa_median = np.std(rmsa_median_all, axis=0)
file_save.write(f"precision:{precision_all}\n")
file_save.write(f"precision, P@1:{mean_precision[0]:.4f}({std_precision[0]:.4f})\n")
file_save.write(f"precision, P@2:{mean_precision[1]:.4f}({std_precision[1]:.4f})\n")
file_save.write(f"precision, P@3:{mean_precision[2]:.4f}({std_precision[2]:.4f})\n")
file_save.write(f"cluster_mean_0:{mean_cluster[0]:.4f}({std_cluster[0]:.4f}),cluster_mean_1:{mean_cluster[1]:.4f}({std_cluster[1]:.4f})\n")
file_save.write(f"rmsa_mean_0:{mean_rmsa[0]:.4f}({std_rmsa[0]:.4f}),rmsa_mean_1:{mean_rmsa[1]:.4f}({std_rmsa[1]:.4f})\n")
file_save.write(f"rmsa_median_mean_0:{mean_rmsa_median[0]:.4f}({std_rmsa_median[0]:.4f}),rmsa_median_mean_1:{mean_rmsa_median[1]:.4f}({std_rmsa_median[1]:.4f})\n")

file_save.close()