import os
import torch.utils.data as data
from PIL import Image
import torch


class DatasetProcessing(data.Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None, flag=False):
        self.img_path = data_path
        self.transform = transform
        self.flag = flag
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        # self.img_filename = [x.split(' ')[0][2:] for x in fp]
        self.img_filename = [x.split(' ')[0] for x in fp]
        fp.close()

        fp = open(img_filepath, 'r')
        # self.skt_filename = [x.split(' ')[1][2:].strip() for x in fp]
        self.skt_filename = [x.split(' ')[1].strip() for x in fp]
        fp.close()

        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels

    def __getitem__(self, index):
        if self.flag == False:
            img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
            skt = Image.open(os.path.join(self.img_path, self.skt_filename[index]))
            img = img.convert('RGB')
            # img = img.convert('L')
            skt = skt.convert('RGB')
            # skt = skt.convert('L')
            # img = np.expand_dims(img,2)
            # skt = np.expand_dims(skt,2)
            # img = np.concatenate((img,img,img),axis=2)
            # skt = np.concatenate((skt,skt,skt),axis=2)
            if self.transform is not None:
                img = self.transform(img)
                skt = self.transform(skt)
            label = torch.LongTensor([self.label[index]])
            A = skt
            B = img
            return A, B, label, index
        elif self.flag == True:
            skt = Image.open(os.path.join(self.img_path, self.skt_filename[index]))
            skt = skt.convert('RGB')
            if self.transform is not None:
                skt = self.transform(skt)
            return skt, index

    def __len__(self):
        return len(self.img_filename)
def get_data_loader(data_path,img_filename,label_filename,transform=None,flag=False, batch_size=32, shuffle=True, num_workers=4,):
    dset = DatasetProcessing(data_path, img_filename, label_filename, transform=transform, flag=False)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                             )
    return data_loader
# collection='Sketchy'
# rootpath='/home/zlm/VisualSearch/GSH'
# DATA_DIR=os.path.join(rootpath,collection,'fine-grained','all')
# TRAIN_FILE =os.path.join(rootpath,collection,'fine-grained','train.txt')
# TEST_FILE =os.path.join(rootpath,collection,'fine-grained','test.txt')
# TRAIN_LABEL=os.path.join(rootpath,collection,'fine-grained','train_label.txt')
# TEST_LABEL=os.path.join(rootpath,collection,'fine-grained','test_label.txt')
# test_data=DatasetProcessing(DATA_DIR,TEST_FILE,TEST_LABEL)
# test_loader=get_data_loader(DATA_DIR,TEST_FILE,TEST_LABEL,transform=transformations)
# #test_loader=data.DataLoader(test_data,batch_size=4,shuffle=False,num_workers=2)
# for iteration, batch in enumerate(test_loader,0):
#     A=batch[2]
#     print(A)