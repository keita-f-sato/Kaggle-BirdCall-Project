import pickle
import logging

import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.distributions import Uniform
from torch.utils.data import DataLoader, Dataset

from torchaudio.transforms import Spectrogram, MelSpectrogram
from torchaudio.transforms import TimeStretch, AmplitudeToDB, ComplexNorm 


BIRD_CODE = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


class LoadTrainDataset(Dataset):
    def __init__(self, 
                 df, 
                 sound_dir, 
                 audio_sec=5,
                 sample_rate=32000
                ):
        
        self.train_df = df
        self.sound_dir = sound_dir
        self.audio_sec = audio_sec
        self.sample_rate = sample_rate
        self.target_lenght = sample_rate * audio_sec
    
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, ix):
        sound_info = self.train_df[ix]
        
        waveform = torch.load(sound_info[0])
        input_audio_lenght = waveform.size(1)
        target = torch.zeros([264], dtype=torch.float32)
        target[sound_info[1].item()] = 1
        
        if input_audio_lenght > self.target_lenght:
            dist = torch.randint(0, input_audio_lenght-self.target_lenght, (1,)).item()
            waveform = waveform[:, dist:dist + self.target_lenght]
        else:
            waveform = torch.cat([waveform, torch.zeros([1, self.target_lenght - input_audio_lenght])], dim=1)
            
        return waveform, target, sound_info[1]        
        
        
class cnn_audio(nn.Module):
    def __init__(self, 
                 output_class=264,
                 d_size=256,
                 sample_rate=32000, 
                 n_fft=2**11, 
                 top_db=80):
        
        super().__init__()
        self.mel = MelSpectrogram(sample_rate, n_fft=n_fft)
        self.norm_db = AmplitudeToDB(top_db=top_db)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=3)
        self.dropout3 = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(12, 256, 2, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        self.bn_lstm = nn.BatchNorm1d(256)
        
        self.output1 = nn.Linear(256, 512)
        self.relu_out = nn.ReLU(0.1)
        self.dropout_out = nn.Dropout(0.3)
        self.output2 = nn.Linear(512, output_class)
    
    def forward(self, x):
        x = self.mel(x)
        x = self.norm_db(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x, _ = self.lstm(x.view(x.size(0), 256, 12), None)
        x = self.dropout_lstm(x[:, -1, :])
        x = self.bn_lstm(x)
        
        x = x.view(-1, 256)
        x = self.output1(x)
        x = self.relu_out(x)
        x = self.dropout_out(x)
        
        x = self.output2(x)
        
        return x
    
def convert_label(predict):
    return [np.argwhere(predict[i] == predict[i].max())[0].item() for i in range(len(predict))]

def get_F1_score(y_true, y_pred, average):
    return f1_score(y_true, y_pred, average=average)
 
class Trainer():
    def __init__(self, train_dataloader, test_dataloader, lr, betas, weight_decay, log_freq, with_cuda, model=None):
        
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        print("Use:", "cuda:0" if cuda_condition else "cpu")
        
        self.model = cnn_audio().to(self.device)
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, 5)
        self.criterion = nn.BCEWithLogitsLoss()
        
        if model != None:            
            checkpoint = torch.load(model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.criterion = checkpoint['loss']


        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        print("Using %d GPUS for Converter" % torch.cuda.device_count())
        
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        self.test_loss = []
        self.train_loss = []
        self.train_f1_score = []
        self.test_f1_score = []
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        str_code = "train" if train else "test"

        data_iter = tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")
        
        total_element = 0
        loss_store = 0.0
        f1_score_store = 0.0
        total_correct = 0

        for i, data in data_iter:
            specgram = data[0].to(self.device)
            label = data[2].to(self.device)
            one_hot_label = data[1].to(self.device)
            predict_label = self.model(specgram)

            # 
            predict_f1_score = get_F1_score(
                label.cpu().detach().numpy(),
                convert_label(predict_label.cpu().detach().numpy()),
                average='micro'
            )
            
            loss = self.criterion(predict_label, one_hot_label)

            # 
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()

            loss_store += loss.item()
            f1_score_store += predict_f1_score
            self.avg_loss = loss_store / (i + 1)
            self.avg_f1_score = f1_score_store / (i + 1)
        
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": round(self.avg_loss, 5),
                "loss": round(loss.item(), 5),
                "avg_f1_score": round(self.avg_f1_score, 5)
            }

        data_iter.write(str(post_fix))
        self.train_loss.append(self.avg_loss) if train else self.test_loss.append(self.avg_loss)
        self.train_f1_score.append(self.avg_f1_score) if train else self.test_f1_score.append(self.avg_f1_score)
        
    
    def save(self, epoch, file_path="../models/2j-drop03/"):
        """
        """
        output_path = file_path + f"crnn_ep{epoch}.model"
        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'criterion': self.criterion
            },
            output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    def export_log(self, epoch, file_path="../../logs/2j-drop03/"):
        df = pd.DataFrame({
            "train_loss": self.train_loss, 
            "test_loss": self.test_loss, 
            "train_F1_score": self.train_f1_score,
            "test_F1_score": self.test_f1_score
        })
        output_path = file_path+f"loss_{epoch}.log"
        print("EP:%d logs Saved on:" % epoch, output_path)
        df.to_csv(output_path)
        
logger = logging.getLogger('ErrorLogging')
 
fh = logging.FileHandler('../../logs/err_log_2jd3.log')
logger.addHandler(fh)
 
sh = logging.StreamHandler()

try:        

    #train_data = pd.read_csv("../../dataset/train.csv")
    #train_data = train_data[train_data["filename"] != 'XC195038.mp3']
    folder = "../../dataset/tensor_audio"

    with open('../../dataset/train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)

    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state = 2)

    train_dataset = LoadTrainDataset(train_data, folder)
    test_dataset = LoadTrainDataset(test_data, folder)

    batch_size = 32
    num_workers= 5

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    lr=1e-3
    weight_decay=0.0001
    adam_beta1=0.5
    adam_beta2=0.99
    betas = (adam_beta1, adam_beta2)

    log_freq=100
    with_cuda=True

    model = None

    trainer = Trainer(train_dataloader, test_dataloader, lr, betas, weight_decay, log_freq, with_cuda, model)


    #%%capture output
    epochs = 1000
    print("Training Start")

    for epoch in range(0, epochs):
        trainer.train(epoch)
        # Model Save
        trainer.test(epoch)
        if epoch % 50 == 0 and epoch != 0:
            trainer.save(epoch)
            trainer.export_log(epoch)
except Exception as err:
    logger.exception('Raise Exception: %s', err)
