import pickle
import logging

import random
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, lr_scheduler
from torch.distributions import Uniform
from torch.utils.data import DataLoader, Dataset

from torchaudio.transforms import Spectrogram, MelSpectrogram
from torchaudio.transforms import TimeStretch, AmplitudeToDB, ComplexNorm, Resample
from torchaudio.transforms import FrequencyMasking, TimeMasking


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
                 sample_rate=32000,
                 max_perc=0.4
                ):
        
        self.train_df = df
        self.sound_dir = sound_dir
        self.audio_sec = audio_sec
        self.sample_rate = sample_rate
        self.target_lenght = sample_rate * audio_sec
        self.max_perc = max_perc
    
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, ix):
        sound_info = self.train_df[ix]
        
        waveform = torch.load(sound_info[0])
        target = torch.zeros([264], dtype=torch.float32)
        target[sound_info[1].item()] = 1
        
        waveform = self.norm_5sec(waveform)
        
        return waveform, target, sound_info[1]    
    
    def norm_5sec(self, waveform):
        input_audio_lenght = waveform.size(1)
        
        if input_audio_lenght > self.target_lenght:
            dist = torch.randint(0, input_audio_lenght-self.target_lenght, (1,)).item()
            waveform = waveform[:, dist:dist + self.target_lenght]
        else:
            waveform = torch.cat([waveform, torch.zeros([1, self.target_lenght - input_audio_lenght])], dim=1)
        
        return waveform
    
    
class RondomStretchMelSpectrogram(nn.Module):
    def __init__(self, sample_rate, n_fft, top_db, max_perc):
        super().__init__()
        self.time_stretch = TimeStretch(hop_length=None, n_freq=n_fft//2+1)
        self.stft = Spectrogram(n_fft=n_fft, power=None)
        self.com_norm = ComplexNorm(power=2.)
        self.fm = FrequencyMasking(100)
        self.tm = TimeMasking(100)
        self.mel_specgram = MelSpectrogram(sample_rate, n_fft=n_fft, f_max=8000)
        self.AtoDB= AmplitudeToDB(top_db=top_db)
        self.max_perc = max_perc
        self.sample_rate = sample_rate
        self.resamples = [
                Resample(sample_rate, sample_rate*0.6),
                Resample(sample_rate, sample_rate*0.7),
                Resample(sample_rate, sample_rate*0.8),
                Resample(sample_rate, sample_rate*0.9),
                Resample(sample_rate, sample_rate*1),
                Resample(sample_rate, sample_rate*1.1),
                Resample(sample_rate, sample_rate*1.2),
                Resample(sample_rate, sample_rate*1.3),
                Resample(sample_rate, sample_rate*1.4)
            ]
    
    def forward(self, x, train):
        x = random.choice(self.resamples)(x)
        
        x = self.stft(x)

        if train:
            dist = Uniform(1.-self.max_perc, 1+self.max_perc)
            x = self.time_stretch(x, dist.sample().item())
            x = self.com_norm(x)
            x = self.fm(x, 0)
            x = self.tm(x, 0)
        else:
            x = self.com_norm(x)
        
        x = self.mel_specgram.mel_scale(x)
        x = self.AtoDB(x)
        
        size = torch.tensor(x.size())
        
        if size[3] > 157:
            x = x[:,:,:,0:157]
        else:
            x = torch.cat([x, torch.cuda.FloatTensor(size[0], size[1], size[2], 157 - size[3]).fill_(0)], dim=3)
        
        return x

    
def convert_label(predict):
    return [np.argwhere(predict[i] == predict[i].max())[0].item() for i in range(len(predict))]

def get_F1_score(y_true, y_pred, average):
    return f1_score(y_true, y_pred, average=average)

def adaptive_concat_pool2d(x, sz=(1,1)):
    out1 = F.adaptive_avg_pool2d(x, sz).view(x.size(0), -1)
    out2 = F.adaptive_max_pool2d(x, sz).view(x.size(0), -1)
    return torch.cat([out1, out2], 1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        super().__init__()
        
        padding = kernel_size // 2
        self.pool = pool
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x): # x.shape = [batch_size, in_channels, a, b]
        x1 = self.conv1(x)
        x = self.conv2(torch.cat([x, x1],1))
        if(self.pool): x = F.avg_pool2d(x, 2)
        return x   # x.shape = [batch_size, out_channels, a//2, b//2]

    
class Classifier_M3(nn.Module):
    def __init__(self, num_classes=264):
        super().__init__()
        self.mel = RondomStretchMelSpectrogram(sample_rate=32_000, n_fft=2**11, top_db=80, max_perc=0.4)
        self.conv1 = ConvBlock(1,64)
        self.conv2 = ConvBlock(64,128)
        self.conv3 = ConvBlock(128,256)
        self.conv4 = ConvBlock(256,512)
        self.conv5 = ConvBlock(512,1024,pool=False)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(3840),
            nn.Linear(3840, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, train): # batch_size, 3, a, b
        x = self.mel(x, train)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        #pyramid pooling
        x = torch.cat([adaptive_concat_pool2d(x2), adaptive_concat_pool2d(x3),
                       adaptive_concat_pool2d(x4),adaptive_concat_pool2d(x5)], 1)
        x = self.fc(x)
        return x
    
class Trainer():
    def __init__(self, train_dataloader, test_dataloader, lr, betas, weight_decay, log_freq, with_cuda, model=None):
        
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        print("Use:", "cuda:0" if cuda_condition else "cpu")
        
        self.model = Classifier_M3().to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
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
            predict_label = self.model(specgram, train)

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
        
    
    def save(self, epoch, file_path="../models/2k/"):
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
    
    def export_log(self, epoch, file_path="../../logs/2k/"):
        df = pd.DataFrame({
            "train_loss": self.train_loss, 
            "test_loss": self.test_loss, 
            "train_F1_score": self.train_f1_score,
            "test_F1_score": self.test_f1_score
        })
        output_path = file_path+f"loss_timestrech.log"
        print("EP:%d logs Saved on:" % epoch, output_path)
        df.to_csv(output_path)
        
        
logger = logging.getLogger('ErrorLogging')
 
fh = logging.FileHandler('../../logs/err_log_2jadamW.log')
logger.addHandler(fh)
 
sh = logging.StreamHandler()

try:        
    folder = "../../dataset/tensor_audio"

    with open('../../dataset/train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)

    with open('../../dataset/test_data.pickle', 'rb') as f:
        test_data = pickle.load(f)


    train_dataset = LoadTrainDataset(train_data, folder)
    test_dataset = LoadTrainDataset(test_data, folder)

    batch_size = 32
    num_workers= 5

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    lr = 1e-3
    weight_decay=0.0
    adam_beta1=0.9
    adam_beta2=0.999
    betas = (adam_beta1, adam_beta2)

    log_freq=100
    with_cuda=True

    model = None

    trainer = Trainer(train_dataloader, test_dataloader, lr, betas, weight_decay, log_freq, with_cuda, model)
    
    epochs = 1000
    print("Training Start")

    for epoch in range(0, epochs):
        trainer.train(epoch)
        # Model Save
        trainer.test(epoch)
        trainer.export_log(epoch)
        if epoch % 50 == 0 and epoch != 0:
            trainer.save(epoch)
    trainer.save(epoch)

except Exception as err:
    logger.exception('Raise Exception: %s', err)
