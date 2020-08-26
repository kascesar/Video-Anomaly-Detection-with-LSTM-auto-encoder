from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import os
from sys import exit


class AnomalyDataGen(Sequence):
    def __init__(self, data_path, win_size=10, batch_size=10, shuffle=True,
                 img_input_shape=(256, 256, 1), aug_steps=None, train=None):
        '''
        :param data_path: str -> path to data
        :param batch_size: Int -> batch size
        :param shufle: Bool -> shufle data after every epoch
        :param to_fit: Bol -> si el generador es para entrenamiento
        :param aug_steps: list -> representa el stride de las ventanas que
                                  contienen cada batche.
                                  Por defecto = [1, 2, 3, 4]
        '''
        self.img_input_shape = img_input_shape
        self.win_size = win_size
        self.batch_size = batch_size
        self.shufle = shuffle
        self.data_path = data_path
        self.suported_formats = ['png', 'jpeg', 'JPEG', 'tif']

        if train is None:
            self.train = True
        else:
            self.train = train

        if self.train:
            self.TT_path = 'Train'
        else:
            self.TT_path = 'Test'

        if aug_steps is None:
            aug_steps = [1, 2, 3, 4]
            self.aug_steps = aug_steps
        else:
            self.aug_steps = aug_steps

        self.train_folder = os.path.join(self.data_path, self.TT_path)
        self.folder_list = os.listdir(self.train_folder)
        self.folder_list.sort()
        self.inds = self.inds_maker(self.train_folder, self.folder_list)
        self.batches_inds = self.inds_batch_maker(inds=self.inds,
                                                  batch_size=self.batch_size,
                                                  win_size=self.win_size,
                                                  aug_steps=self.aug_steps)
        self.on_epoch_end()

    def __len__(self):
        return len(self.batches_inds)

    def on_epoch_end(self):
        """Randomize de batches"""
        if self.shufle:
            np.random.shuffle(self.batches_inds)

    def inds_maker(self, train_folder, folder_list):
        inds = []
        folder_inds = range(len(folder_list))
        for f_ind in folder_inds:
            i = [f_ind, range(len(os.listdir(
                                  os.path.join(train_folder,
                                               folder_list[f_ind]))
                                  )
                              )
                 ]

            inds.append(i)
        return inds

    def inds_batch_maker(self, inds, batch_size, win_size, aug_steps):
        windows_inds = []
        for step in aug_steps:
            for folder_inds, files_inds in inds:
                j_i, j_n = files_inds[0], files_inds[-1]
                indexes = np.arange(j_i, j_n, step, dtype=np.int)
                cont = 0
                while True:
                    win_idxs = indexes[cont: win_size + cont]
                    if len(win_idxs) != win_size:
                        break
                    cont += 1
                    windows_inds.append([folder_inds, win_idxs])

        windows_inds = np.array(windows_inds)
        folder_inds = set(list(windows_inds[:, 0]))
        batches_inds = []
        for i in folder_inds:
            a = windows_inds[windows_inds[:, 0] == i]
            b = np.arange(0, len(a), batch_size, dtype=np.int)
            for j in b:
                batches_inds.append(a[j: j + batch_size])
        return batches_inds

    def imgs_path2tensor(self, name_list):
        win = None
        for name in name_list:
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_input_shape[0],
                                   self.img_input_shape[1])
                             )
            img = img / 256.0
            img = np.reshape(img, (1, *self.img_input_shape))
            if img is None:
                print('asdasd')
                exit()
            if win is None:
                win = img
            else:
                win = np.vstack((win, img))
        return win

    def make_Xy_batch(self, batch_inds):
        # batch_size = len(batch_inds)
        folders_inds = batch_inds[:, 0]
        folders_inds = list(set(folders_inds))

        # Chek if the inds_batch is correcto
        if len(folders_inds) != 1:
            print('batches de imagenes combinadas, algo anda mal!!!')
            exit()

        batch = []
        for folder_id, files_ids in batch_inds:

            # generate a list path to the img for this temporal-window
            folder_name = self.folder_list[folder_id]
            folder_path = os.path.join(self.train_folder, folder_name)
            # list file and sort them, to make a correct temporal-window
            imgs_names = os.listdir(folder_path)
            imgs_names.sort()

            all_imgs_paths = []
            for name in imgs_names:
                img_path = os.path.join(folder_path, name)

                if not os.path.isfile(img_path):
                    print('El archivo no existe ... saliendo')
                    exit()
                all_imgs_paths.append(img_path)
            imgs_paths = [all_imgs_paths[i] for i in files_ids]

            window_imgs = self.imgs_path2tensor(name_list=imgs_paths)

            # make a tensoir with n-images
            batch.append(window_imgs)

        X = np.array(batch, dtype=np.float32)
        if X is None:
            print('X es None... error')
            exit()

        return X, X

    def __getitem__(self, index):
        batch_inds = self.batches_inds[index]
        batch_inds = np.array(batch_inds)
        out = self.make_Xy_batch(batch_inds=batch_inds)
        if self.train:
            return out
        else:
            return out[0]
