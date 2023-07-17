import torch
import numpy as np
from tqdm import tqdm
import h5py
import sinkhornknopp as sk
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from os.path import dirname, abspath, join, exists
from os import makedirs
from datetime import datetime
import json
import os
from kcenter_greedy import kCenterGreedy
from sampling_def import SamplingMethod
from scipy.optimize import linear_sum_assignment
import scipy

PAD_INDEX = 0

BASE_DIR = dirname(abspath(__file__))

def do_spectral_clustering(latent, num_clusters, k):
    l = 5e1
    mu = 1.
    u,s,v = np.linalg.svd(mu*np.eye(latent.shape[1]) + l*np.matmul(latent.T, latent), full_matrices = False)
    H_inv = np.matmul(v[:k].T, (u[:,:k]/np.expand_dims(s[:k],0)).T)
    A_latent = l*np.matmul(np.matmul(latent, H_inv), latent.T)
    A_latent = np.maximum(0, A_latent)
    clustering = SpectralClustering(n_clusters = num_clusters, affinity = 'precomputed')
    prediction = clustering.fit(A_latent)
    print ('Optimal A is computed')

    results = prediction.labels_
    return results



def prepare_data(num_days, features_used, user_name):
    if features_used == 'CNN':
        hdr = h5py.File('', 'r')
    if features_used == 'LSTM':
        # hdr = h5py.File('/home/pravinn/dataset/features/ADL_features_lstm.h5', 'r')
        hdr = h5py.File('/fs/cfar-projects/pravin/recovering_activities_patterns/Egoroutine/ADL_features_lstm.h5', 'r')

    user1_days = hdr[user_name].keys()
    test_images = 'None'
    for i, val in enumerate(user1_days):
        if i < num_days:
            feat = hdr[user_name + '/' +val+ '/features']
            if test_images == 'None':
                test_images = feat
            else:
                test_images = np.concatenate((test_images, feat), axis = 0)
    # test_images = test_images[0:20480,:]
    print (np.max(test_images), np.min(test_images))
    # test_images = np.expand_dims(test_images, 0)
    return test_images

# def get_euclidean_distance(A_matrix, B_matrix):
#     """
#     Function computes euclidean distance between matrix A and B.
#     E. g. C[2,15] is distance between point 2 from A (A[2]) matrix and point 15 from matrix B (B[15])
#     Args:
#         A_matrix (numpy.ndarray): Matrix size N1:D
#         B_matrix (numpy.ndarray): Matrix size N2:D
#     Returns:
#         numpy.ndarray: Matrix size N1:N2
#     """
#
#     A_square = np.reshape(np.sum(A_matrix * A_matrix, axis=1), (A_matrix.shape[0], 1))
#     B_square = np.reshape(np.sum(B_matrix * B_matrix, axis=1), (1, B_matrix.shape[0]))
#     AB = A_matrix @ B_matrix.T
#
#     C = -2 * AB + B_square + A_square
#
#     return np.sqrt(C)


def get_labels(A_matrix, centroids):
    """
    Function computes euclidean distance between matrix A and B.
    E. g. C[2,15] is distance between point 2 from A (A[2]) matrix and point 15 from matrix B (B[15])
    Args:
        A_matrix (numpy.ndarray): Matrix size N1:D
        B_matrix (numpy.ndarray): Matrix size N2:D
    Returns:coreset
        numpy.ndarray: Matrix size N1:N2
    """
    A_matrix = np.squeeze(A_matrix)
    B_matrix = A_matrix[centroids,:]
    # print (A_matrix.shape, B_matrix.shape)

    A_square = np.reshape(np.sum(A_matrix * A_matrix, axis=1), (A_matrix.shape[0], 1))
    B_square = np.reshape(np.sum(B_matrix * B_matrix, axis=1), (1, B_matrix.shape[0]))
    AB = A_matrix @ B_matrix.T

    C = -2 * AB + B_square + A_square

    return np.argmin(np.sqrt(C), axis=1)

class EpochSeq2SeqTrainer:

    def __init__(self, model,
                 loss_function, optimizer,
                 logger, run_name,
                 save_config, save_checkpoint,
                 config):

        features_used = 'LSTM' # featreus = CNN/LSTM
        day_mapping = { 'user_01':14,'user_02':10,'user_03':16,'user_04':20,'user_05':13,'user_06':18,'user_07':13}

        self.user_name = 'user_02'
        num_days =  day_mapping[self.user_name]

        self.do_test = 50
        self.do_optimiztion = 50
        self.config = config
        self.device = torch.device(self.config['device'])

        self.model = model.to(self.device)
        self.model= torch.nn.DataParallel(self.model)

        self.train_data = prepare_data(num_days, features_used, self.user_name)
        ## PCA code
        pca = PCA(n_components=512)
        print (self.train_data.shape)
        print ("PCA...")
        pca.fit(self.train_data)
        self.train_data = pca.transform(self.train_data)
        print (self.train_data.shape)
        self.train_data = np.expand_dims(self.train_data, 0)
        
        self.coreset =  kCenterGreedy(None, None, 0)

        self.loss_function = loss_function.to(self.device)
        self.optimizer = optimizer
        self.clip_grads = self.config['clip_grads']
        self.num_clusters = self.config['num_clusters']
        self.d_ff= self.config['d_ff']
        self.heads_count = self.config['heads_count']
        self.out_dir = self.config['out_dir']
        self.num_of_layers = self.config['layers_count']


        print ('\n')
        print ("------------------------")
        print ("------------------------")
        print ('Processing {} with number of  days {}'.format(self.user_name, num_days))
        print ('Number of clusters for selflabeling loss are {}'.format(self.num_clusters))
        print ("------------------------")
        print ("------------------------")
        print ('\n')

        self.logger = logger
        self.checkpoint_dir = join(BASE_DIR, 'checkpoints', run_name)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        if save_config is None:
            config_filepath = join(self.checkpoint_dir, 'config.json')
        else:
            config_filepath = save_config
        with open(config_filepath, 'w') as config_file:
            json.dump(self.config, config_file)

        self.print_every = self.config['print_every']
        self.save_every = self.config['save_every']

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

        self.save_checkpoint = save_checkpoint
        self.save_format = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth'

        self.log_format = (
            "Epoch: {epoch:>3} "
            "Progress: {progress:<.1%} "
            "Elapsed: {elapsed} "
            "Examples/second: {per_second:<.1} "
            "Train Loss: {train_loss:<.6} "
            "Val Loss: {val_loss:<.6} "
            "Train Metrics: {train_metrics} "
            "Val Metrics: {val_metrics} "
            "Learning rate: {current_lr:<.4} "
        )

    def run_epoch(self, targets,  mode='train'):
        emb, yhat, PS, rep_loss = self.model(self.train_data) # (batch_size, seq_len, d_model)
        # print ("two losses", self.loss_function(yhat, targets), rep_loss)
        loss = self.loss_function(yhat, targets)+ rep_loss

        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

        return loss, emb, yhat, PS


    def run(self, max_epochs=10):
        for epoch in range(self.epoch, max_epochs + 1):
            if epoch == 0:
                # initiate labels as shuffled.
                centroids = self.coreset.select_batch_(self.train_data, [], self.num_clusters)
                L = get_labels(self.train_data, centroids)
                L = np.reshape(L, (1, L.shape[0]))
                self.train_data = torch.from_numpy(self.train_data)

            if torch.cuda.is_available():
                targets = torch.from_numpy(L[0]).long().cuda()
                # self.train_data = self.train_data.cuda()
            else:
                targets = torch.from_numpy(L[0]).long()


            self.epoch = epoch

            self.model.train()

            epoch_start_time = datetime.now()
            loss, _, _ ,_= self.run_epoch(targets, mode='train')
            epoch_end_time = datetime.now()

            #from collections import Counter
            #cnt = Counter(L[0])
            #cnt = cnt.most_common()
            #print (cnt)

            if epoch%self.do_optimiztion==0:
                self.model.eval()
                _, emb, _, _= self.run_epoch(targets, mode='val')
                emb = torch.squeeze(emb)
                centroids_new = self.coreset.select_batch_(emb.detach().cpu().numpy(), [], self.num_clusters)
                dist = scipy.spatial.distance_matrix(emb[centroids,:].detach().cpu().numpy(), emb[centroids_new,:].detach().cpu().numpy())
                row_ind, col_ind = linear_sum_assignment(dist)
                centroids_new_mapped = [centroids_new[ii_idx] for ii_idx in col_ind]


                # print('opt took {0:.2f}min,'.format((time.time() - tt)), flush=True)
                np.save(self.out_dir + 'output_Self_Labels.npy', L)
                # code to map the new clusters with the old ones
                L_new = get_labels(emb.detach().cpu().numpy(), centroids_new_mapped)

                L= np.reshape(L_new, (1, L_new.shape[0]))
                centroids = centroids_new


            if epoch%self.do_test==0 or epoch == max_epochs-1:
                self.model.eval()
                loss, emb, _, PS = self.run_epoch(targets, mode='val')
                results = do_spectral_clustering(emb.detach().cpu().numpy(), self.num_clusters, 90)
                template = 'Global step {0:5d}: loss = {1:0.4f} \n'
                print (template.format(epoch, loss.data))
                np.save(self.out_dir + self.user_name + '_rep_projection_256_rep_loss_sharedQK_performer_coreset_results_epoch_' + str(epoch) + '_layers_' + str(self.num_of_layers) +'_headcount_' + str(self.heads_count)  + '_feedforward_' + str(self.d_ff)  +'_clusters_' + str(self.num_clusters)  + '.npy', results)
                np.save(self.out_dir + self.user_name + '_rep_projection_256_rep_loss_sharedQK_performer_coreset_latent_epoch_' + str(epoch) + '_layers_' + str(self.num_of_layers) + '_headcount_' + str(self.heads_count)  +'_feedforward_' + str(self.d_ff) +  '_clusters_' + str(self.num_clusters)  +'.npy', emb.detach().cpu().numpy())
                # np.save(self.out_dir + self.user_name + '_performer_coreset_attn_weights_epoch_' + str(epoch) + '_headcount_' + str(self.heads_count)  + '_feedforward_' + str(self.d_ff) + '_clusters_' + str(self.num_clusters)  + '.npy', attn_weights.detach().cpu().numpy())
                np.save(self.out_dir + self.user_name + '_rep_projection_256_rep_loss_sharedQK_performer_coreset_PS_epoch_' + str(epoch) + '_layers_' + str(self.num_of_layers) + '_headcount_' + str(self.heads_count)  + '_feedforward_' + str(self.d_ff) + '_clusters_' + str(self.num_clusters)  + '.npy', PS.detach().cpu().numpy())

            # if epoch % self.print_every == 0 and self.logger:
            #     per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
            #     current_lr = self.optimizer.param_groups[0]['lr']
            #     log_message = self.log_format.format(epoch=epoch,
            #                                          progress=epoch / epochs,
            #                                          per_second=per_second,
            #                                          train_loss=train_epoch_loss,
            #                                          val_loss=val_epoch_loss,
            #                                          train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
            #                                          val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
            #                                          current_lr=current_lr,
            #                                          elapsed=self._elapsed_time()
            #                                          )
            #
            #     self.logger.info(log_message)
            #
            # if epoch % self.save_every == 0:
            #     self._save_model(epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics)

    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics):

        checkpoint_filename = self.save_format.format(
            epoch=epoch,
            val_loss=val_epoch_loss,
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        if self.save_checkpoint is None:
            checkpoint_filepath = join(self.checkpoint_dir, checkpoint_filename)
        else:
            checkpoint_filepath = self.save_checkpoint

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'train_metrics': train_epoch_metrics,
            'val_loss': val_epoch_loss,
            'val_metrics': val_epoch_metrics,
            'checkpoint': checkpoint_filepath,
        }

        if self.epoch > 0:
            torch.save(self.model.state_dict(), checkpoint_filepath)
            self.history.append(save_state)

        representative_val_metric = val_epoch_metrics[0]
        if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_loss_at_best = val_epoch_loss
            self.train_loss_at_best = train_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.val_metrics_at_best = val_epoch_metrics
            self.best_checkpoint_filepath = checkpoint_filepath

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds
