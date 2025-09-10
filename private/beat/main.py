#!/usr/bin/env python3
"""
Beat Modulation

Symbol detection via pretrained ML decoder.

TODO
----
* add read/write to specific filename
* symbol syncronization
* balance my dataset? it's kinda balanced...
* add symbol edge marker &target
* add weight saving .pt

Record
------
weights, optimizer,       epochs, val_loss, name, notes
pretrain, madgrad, 10+10+205=225, 7.53e-6, step4, good model
pretrain,    adam,  30+30+70=130, 1.03e-4, version0, maybe not enough patience
pretrain, madgrad,  30+30+44=104, 0.00545, version1, hmm not great
pretrain,    adam,            34, 0.00196, adam-scratch, converged w/o curriculum training very well
pretrain, madgrad,            55, 0.00529, converged w/o curriculum training very well
no-pretrain, adam,      30+40=70, 0.00227, version2, pretty good
no-pretrain, madgrad, 30+287=117,       0, version3, 100% accuracy
no-pretrain, madgrad, 30+30+114=174, 0.00052, allsym, space detect not working


Notes
-----
* Pretrained weights negligible impact
"""
import argparse
import logging
import os
import sys
import tempfile  # cross-platform way to get temp directory
import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torchmetrics
from einops import rearrange
from madgrad import MADGRAD
from rich.logging import RichHandler
from scipy.signal import chirp, firwin, lfilter, resample_poly
from scipy.signal.windows import gaussian, hamming, tukey
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchinfo import summary
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)
from torchvision.models.resnet import BasicBlock, ResNet, resnet18

FORMAT = "%(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")
logging.getLogger("matplotlib.font_manager").disabled = True

# validate LUFS with "ffmpeg -i x.wav -af loudnorm=I=-14:print_format=summary -f null -"
AUDIO_RMS = 0.15  # peak amplitude for -14 LUFS
TMP_PATH = Path(tempfile.gettempdir())
BEAT_PATH = Path("~/Music/20syl/36 (Beats & Types) EP").expanduser()
CORPUS_CACHE = TMP_PATH / "corpus.pkl"


def rolling_rms(ray, num_samps: int):
    """calculate the rolling RMS for input signal over some window"""
    xc = np.cumsum(abs(ray) ** 2)
    rms = np.sqrt((xc[num_samps:] - xc[:-num_samps]) / num_samps)
    # stack some extra zeros at the end to match length, region too small to sample anyway
    return np.hstack((rms, np.zeros(num_samps)))


def interferer(slice_length: int):
    """return a real-valued rms-normalized interferer"""
    events = torch.rand(11)
    ray = events[7] * torch.randn(slice_length, dtype=torch.float32)
    ttt = np.arange(slice_length)
    if events[0] > 0.5:
        # add chirp
        ray += events[3] * chirp(ttt, f0=events[1], f1=events[2], t1=slice_length)
    if events[4] > 0.5:
        # add tone
        ray += events[5] * np.cos(6.283 * events[6] * ttt)
    if events[8] > 0.5:
        # add tone
        ray += events[9] * np.cos(6.283 * events[10] * ttt)
    ray /= torch.sqrt(torch.mean(abs(ray) ** 2))
    return ray


class RecordedDataset(Dataset):
    def __init__(
        self,
        audio_path: str | os.PathLike,
        slice_length: int,
        data_format: str = "nchw",
        fftsize: int = 256,
        stepsize=1,
    ):
        self.samples, _ = soundfile.read(audio_path)
        self.samples = torch.from_numpy(self.samples).type(torch.float32)
        self.num_samples = len(self.samples)
        self.slice_length = slice_length
        self.fftsize = fftsize  # must be divisble by 4
        # calculate hop length to get somewhat square output
        self.stft_hop = round((slice_length - fftsize) / (fftsize // 2 + 1))
        self.stft_window = torch.hann_window(fftsize)
        self.stepsize = stepsize
        self.data_format = data_format
        log.info(f"RecordedDataset({audio_path.stem})")

    def __getitem__(self, idx):
        x_data = self.samples[idx * self.stepsize : idx * self.stepsize + self.slice_length]
        # normalize amplitude back to where the model is comfortable
        rms = torch.sqrt(torch.mean(x_data.abs() ** 2))
        x_data = x_data / rms * AUDIO_RMS

        if self.data_format == "nchw":
            # 2D spectrogram
            x_data = torch.stft(x_data, self.fftsize, hop_length=self.stft_hop, return_complex=True, window=self.stft_window).abs().unsqueeze(0)
        elif self.data_format == "ncl":
            # 1D samples
            x_data = x_data.unsqueeze(0)
        else:
            raise ValueError(f"{self.data_format} not allowed")
        return x_data

    def __len__(self):
        return (self.num_samples - self.slice_length) // self.stepsize


class BeatDataset(Dataset):
    def __init__(
        self,
        data_dir: str | os.PathLike,
        slice_length: int,
        selection: str = "train",
        fftsize: int = 256,
        add_space: bool = True,
        add_noise: bool = True,
        data_format: str = "nchw",
        transform=None,
    ):

        self.file_samp_rate = None
        self.samp_rate = None
        self.data_format = data_format
        self.slice_length = slice_length
        self.selection = f"{selection}_indices"
        self.fftsize = fftsize  # must be even
        # calculate hop length to get somewhat square output
        self.stft_hop = round((slice_length - fftsize) / (fftsize // 2 + 1))
        self.stft_window = torch.hann_window(fftsize)
        self.val_test_snr_db = 5
        self.add_noise = add_noise
        self.set_easy_train_snr_db()
        # load corpus if available
        if CORPUS_CACHE.exists():
            self.corpus = torch.load(CORPUS_CACHE)
            self.samp_rate = self.corpus[0]["sample_rate"]
            corpus_size = CORPUS_CACHE.stat().st_size // 2**20
            log.info(f"read {corpus_size} MB from {CORPUS_CACHE} cache")
        else:
            # read everything
            self.acc = []  # approximate unique slices in each file
            self.corpus = []
            for path in data_dir.glob("*"):
                label, samps, rms = self.read_soundfile(path)
                indices = self.no_split(samps, rms)
                # indices = self.split(samps, rms)
                self.corpus += [{
                    "label" : label,
                    "samples" : samps,
                    "rms" : rms,
                    "sample_rate" : self.samp_rate,
                    "train_indices" : indices[0],
                    "val_indices" : indices[1],
                    "test_indices" : indices[2],
                }]
            if add_space:
                # add a "silent" space character
                self.add_space_symbol()
            torch.save(self.corpus, CORPUS_CACHE)
            corpus_size = CORPUS_CACHE.stat().st_size // 2**20
            log.info(f"wrote {corpus_size} MB to {CORPUS_CACHE} cache")
        # finish reading corpus; we can't precompute acc since it depends on selection
        acc = []
        self.labels = []
        for cdx in range(len(self.corpus)):
            acc += [len(self.corpus[cdx][self.selection]) // slice_length]
            self.labels += [self.corpus[cdx]["label"]]
        if acc[-1] == 0:
            # to balance the dataset during training we need to pretend there are more 'space' samples
            acc[-1] = round(np.mean(acc[:-1]))
        self.acc = acc
        self.acc_sum = np.sum(acc)
        self.acc_cumsum = np.cumsum(acc)
        self.num_classes = len(self.labels)
        self.transform = transform
        all_labels = "".join(label for label in sorted(self.labels))
        log.info(f"BeatDataset({self.acc_sum} {selection} slices of '{all_labels}')")

    def add_space_symbol(self):
        self.corpus += [{
            "label" : "_",
            # amount could be slice length, but need extra for potential offset
            "samples" : torch.zeros(self.slice_length * 2, dtype=torch.float32),
            "rms" : torch.ones(1, dtype=torch.float32),
            "sample_rate" : self.samp_rate,
            "train_indices" : torch.zeros(1, dtype=torch.int32),
            "val_indices"  : torch.zeros(1, dtype=torch.int32),
            "test_indices": torch.zeros(1, dtype=torch.int32),
        }]
        log.info(f"added blank symbol")

    def read_soundfile(self, audio_path, decimation: int = 5):
        """
        For each audio file assuming formatted as "{track:2d} {title}.{ext}"
        * read samples
        * low pass filter
        * calculate RMS
        * cut into 100 sections for later (train, val, test) split

        Returns
        -------
        time_series
        rms
        """
        # single character label
        label = audio_path.stem.split(" ")[1][0].lower()
        # read gives shape (length, channels)
        audio_raw, file_samp_rate = soundfile.read(audio_path)
        if not self.samp_rate:
            # use first file as reference
            self.file_samp_rate = file_samp_rate
            self.samp_rate = file_samp_rate // decimation
        else:
            if self.file_samp_rate != file_samp_rate:
                raise ValueError("different sample rates between files")
        audio_left = audio_raw[:, 0]
        # spec(audio_left)

        # self.taps = firwin(numtaps=131, cutoff=8e3, fs=samp_rate, pass_zero=True)  # 8 KHz LPF
        taps = firwin(numtaps=131, cutoff=[50, 4e3], fs=self.samp_rate, pass_zero=False)  # SSB 4KHz BPF
        audio_filtered = np.convolve(audio_left, taps, mode="same")
        audio_resamp = resample_poly(audio_filtered, up=1, down=decimation)
        # spec(audio_lpf)
        # print('dubg', len(audio_left), audio_left[0:16], max(audio_left))
        # plt.plot(audio_raw[:,0])

        rms = rolling_rms(audio_resamp, self.slice_length)
        if False:
            is_strong = rms > (max(rms) * 0.1)
            plt.plot(audio_lpf, lw=0.4)
            plt.axvline(44100)
            plt.axvline(44100 + 4096)
            plt.plot(rms)
            plt.axhline(max(rms) * 0.1, ls="dashed")
            plt.plot(is_strong)
            plt.show()
        audio_tensor = torch.from_numpy(audio_resamp.astype(np.float32))
        rms_tensor = torch.from_numpy(rms.astype(np.float32))
        log.info(f"read {label} from {audio_path.suffix} resamp to {self.samp_rate} Hz")
        return label, audio_tensor, rms_tensor

    @staticmethod
    def no_split(samps, rms, rms_thresh=0.25):
        """
        all data is training set
        """
        # compute valid start positions with strong signals
        # consider signals valid if in top quantiles
        mask = rms > rms.quantile(rms_thresh)
        valid_indices = torch.arange(len(samps), dtype=torch.int32)[mask]
        return valid_indices, valid_indices, valid_indices

    @staticmethod
    def split(samps, rms, folds_train=16, folds_val=2, folds_test=2, rms_thresh=0.5):
        """
        determine train, val, test regions exclusively where rms is strong

        This is a bit tricky since we need to fold continous regions.

        Example
        -------
        >>> BeatDataset.split(torch.randn(16), torch.arange(16), 3, 1, 1, .1)
        (tensor([ 2,  3, 10, 11,  6,  7], dtype=torch.int32), tensor([8, 9], dtype=torch.int32), tensor([4, 5], dtype=torch.int32))
        """
        # important since we want everyone to be training on the same portions
        torch.manual_seed(0x5CA1EAB1E)
        # valid start positions with strong signals
        mask = rms > (rms.max() * rms_thresh)
        all_indices = torch.arange(len(samps), dtype=torch.int32)[mask]
        folds_total = folds_train + folds_val + folds_test
        folds_shuffle = torch.randperm(folds_total)
        fold_size = len(all_indices) // folds_total

        indices_train = torch.empty(0, dtype=torch.int32)
        for fdx in range(folds_train):
            fold_start = folds_shuffle[fdx] * fold_size
            fold_stop = fold_start + fold_size
            indices_train = torch.cat((indices_train, all_indices[fold_start:fold_stop]))

        indices_val = torch.empty(0, dtype=torch.int32)
        for fdx in range(folds_train, folds_train + folds_val):
            fold_start = folds_shuffle[fdx] * fold_size
            fold_stop = fold_start + fold_size
            indices_val = torch.cat((indices_val, all_indices[fold_start:fold_stop]))

        indices_test = torch.empty(0, dtype=torch.int32)
        for fdx in range(folds_train + folds_val, folds_total):
            fold_start = folds_shuffle[fdx] * fold_size
            fold_stop = fold_start + fold_size
            indices_test = torch.cat((indices_test, all_indices[fold_start:fold_stop]))

        # for indices in [indices_train, indices_val, indices_test]:
        #     # permute everything!
        #     perm = torch.randperm(indices.size)
        #     indices = indices[perm]

        # log.debug(f"train/val/test split ({folds_train}, {folds_val}, {folds_test})")
        return indices_train, indices_val, indices_test

    def generate(self, message: str, target=None):
        """generate a combined waveform given a string"""
        combined_data = torch.zeros(len(message) * self.slice_length + self.fftsize // 2, dtype=torch.float32)
        # check if this is a valid string
        for mdx, char in enumerate(message):
            try:
                cdx = self.labels.index(char)
            except ValueError:
                log.error(f"generate string must be drawn from {''.join(label for label in sorted(self.labels))}")
                raise

            # random sample position
            sdx = np.random.randint(0, len(self.corpus[cdx][self.selection]))
            x_data = self.get_burst(cdx, sdx, padding=self.fftsize // 4)
            # for padded region, apply edges of window
            x_data[: self.fftsize // 2] *= self.stft_window[: self.fftsize // 2]
            x_data[-self.fftsize // 2 :] *= self.stft_window[-self.fftsize // 2 :]
            # stripe into result
            alpha = mdx * self.slice_length
            omega = (mdx + 1) * self.slice_length + self.fftsize // 2
            combined_data[alpha:omega] = x_data
        # trim edges
        combined_data = combined_data[self.fftsize // 4 : -self.fftsize // 4]

        if target is not None:
            # write to a path
            log.info(f"wrote {len(message)} symbols to {target}")
            soundfile.write(target, combined_data, self.samp_rate)
        return combined_data

    def get_burst(self, cdx: int, sdx: int, padding: int = 0):
        """
        retrieve slice from valid slice positions
        optional padding extends both left and right edges for windowing
        """
        if self.corpus[cdx]["label"] == "_":
            # special handling for space symbol
            offset = padding
            rms = 1
        else:
            # calculate offset into file
            offset = self.corpus[cdx][self.selection][sdx]
            if padding:
                # when padding needed, ensure we don't exceed memory boundaries
                offset = max(padding, offset)
            # retrieve precalculated rms
            rms = self.corpus[cdx]["rms"][offset]
        # read and scale samples
        x_data = self.corpus[cdx]["samples"][offset - padding : offset + self.slice_length + padding] / rms * AUDIO_RMS
        return x_data

    def set_easy_train_snr_db(self):
        self.snr_dist = torch.distributions.Beta(3, 3)

    def set_hard_train_snr_db(self):
        self.snr_dist = torch.distributions.Beta(2, 3)

    def sample_train_snr_db(self) -> float:
        """beta is on interval (0,1) and we want SNRs on interval (-10, 50)"""
        return self.snr_dist.sample() * 50 - 10

    def __getitem__(self, idx: int):
        """given a label, return a beat tensor from a possible precomputed position"""
        # which file is this index in? maybe there is a more elegant way...
        cdx = 0
        while idx >= self.acc_cumsum[cdx]:
            cdx += 1

        # samples will return normalized
        if self.selection == "train_indices":
            # random sample position
            sdx = np.random.randint(0, len(self.corpus[cdx][self.selection]))
            x_data = self.get_burst(cdx, sdx)
        else:
            # uniform steps across corpus
            if cdx == 0:
                sdx = idx * self.slice_length
            else:
                sdx = (idx - self.acc_cumsum[cdx - 1]) * self.slice_length
            x_data = self.get_burst(cdx, sdx)

        if self.transform:
            # impairments here
            x_data = self.transform(x_data)

        if self.add_noise:
            # Add noise but keep scale
            if self.selection == "train_indices":
                snr_db = self.sample_train_snr_db()
            else:
                # use fixed dB value
                snr_db = self.val_test_snr_db
            # tricky scaling for signal & noise; recall snr_db = 20*log10(s_rms / n_rms); x_data has amplitude AUDIO_RMS
            snr_rms = 10 ** (snr_db / 20)
            signal_rescale = snr_rms / (snr_rms + 1)
            noise_scale = AUDIO_RMS / (snr_rms + 1)
            x_data = x_data * signal_rescale + torch.randn_like(x_data) * noise_scale

        if self.data_format == "nchw":
            # 2D spectrogram
            x_data = torch.stft(x_data, self.fftsize, hop_length=self.stft_hop, return_complex=True, window=self.stft_window).abs().unsqueeze(0)
        elif self.data_format == "ncl":
            # 1D samples
            x_data = x_data.unsqueeze(0)
        else:
            raise ValueError(f"{self.data_format} not allowed")

        return x_data, cdx

    def __len__(self) -> int:
        return self.acc_sum


class BeatDatamodule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | os.PathLike = BEAT_PATH,
        slice_length: int = 2000,
        samp_rate: int = 8820,
        batch_size: int = 64,
        num_workers: int = 2,
        add_space: bool = True,
        add_noise: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = BEAT_PATH
        self.slice_length = slice_length
        self.samp_rate = samp_rate
        self.add_space = add_space
        self.add_noise = add_noise

        duration = slice_length / samp_rate
        log.info(f"BeatDatamodule(slice_duration={duration:.2} sec)")

    def setup(self, stage: str = None):
        """
        apply splits
        run on each device if distributed
        """
        self.data_train = BeatDataset(self.data_dir, self.slice_length, "train", data_format="nchw", add_space=self.add_space, add_noise=self.add_noise)
        self.data_val = BeatDataset(self.data_dir, self.slice_length, "val", data_format="nchw", add_space=self.add_space, add_noise=self.add_noise)
        self.data_test = BeatDataset(self.data_dir, self.slice_length, "test", data_format="nchw", add_space=self.add_space, add_noise=self.add_noise)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)


class SymbolDetector(L.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.save_hyperparameters()  # saves init values to checkpoints
        self.num_classes = num_classes
        self.project = torch.nn.Conv2d(1, 3, kernel_size=(1, 1), bias=False)
        self.submodel = mobilenet_v3_small(num_classes=num_classes)
        # use pretrained weights
        # self.submodel = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights)
        # self.submodel.classifier[3] = torch.nn.Linear(1024, num_classes, bias=True)

        # overrite final layer for our custom # of classes
        if False:
            self.submodel = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
            # override initial layer so we can consume a single channel
            self.submodel.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 1), stride=2, padding=3, bias=False)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def freeze_submodel(self, requires_grad: bool = False):
        """freeze the feature extractor weights for transfer learning"""
        for param in self.submodel.features.parameters():
            param.requires_grad = requires_grad

    def forward(self, xxx):
        xxx = self.project(xxx)
        xxx = self.submodel(xxx)
        return xxx

    def step(self, batch):
        xxx, yyy = batch
        logits = self.forward(xxx)
        yyy_one_hot = one_hot(yyy, num_classes=self.num_classes).type(torch.float32)
        return logits, yyy_one_hot

    def training_step(self, batch, batch_idx):
        logits, yyy_one_hot = self.step(batch)
        loss = self.criterion(logits, yyy_one_hot)
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, yyy_one_hot = self.step(batch)
        loss = self.criterion(logits, yyy_one_hot)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits, yyy_one_hot = self.step(batch)
        loss = self.criterion(logits, yyy_one_hot)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        # Adam is effective, but madgrad can get you there faster with more noisy results.
        # return torch.optim.Adam(self.parameters(), lr=1e-3)
        return MADGRAD(self.parameters(), lr=1e-2)  # madgrad needs 10x adam LR


def create_callbacks_loggers(lesson: int) -> dict:
    """seem to need new logger for every curricula"""
    ret = {}
    ret["callbacks"] = [
        L.pytorch.callbacks.RichProgressBar(),
        L.pytorch.callbacks.EarlyStopping("val_loss", patience=50),
        L.pytorch.callbacks.ModelCheckpoint(filename="sb-{epoch}-{val_loss:.5f}", monitor="val_loss"),
    ]
    ret["logger"] = L.pytorch.loggers.TensorBoardLogger("lightning_logs", name=f"lesson_{lesson}")
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train symbol detector", action="store_true")
    parser.add_argument("--eval", help="eval symbol detector", action="store_true")
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--write", help="write symbols to file")
    parser.add_argument("--slice-length", type=int, default=44100 // 5 // 2)

    args = parser.parse_args()
    log.debug(args)

    ckpt_best = None
    # ckpt_best = "sb-val_loss=0.001.ckpt" # 4-stage curr pretrain 0dB 38.2%
    # ckpt_best = "sb-val_loss=0.00001.ckpt" # 2-stage curr scratch 0dB 11.05%
    # ckpt_best = "lightning_logs/dlesson_2/version_4/checkpoints/sb-epoch=114-val_loss=0.00035.ckpt" # 0dB 11.8%
    # ckpt_best = "lightning_logs/dlesson_2/version_3/checkpoints/sb-epoch=237-val_loss=0.00001.ckpt" # 0dB 11.4%
    # 0dB 11.15%
    # ckpt_best = 'allsym_0db=22.29.ckpt' # 0db 22.29%
    ckpt_best = 'allsym_0db=18.82.ckpt'

    if args.write:
        ds = BeatDataset(BEAT_PATH, args.slice_length, "train")
        # message = "abcd" * 10
        # message = "the&quick&brown&fox&jumped&over&the&lazy&dog" * 4
        _ = ds.generate(args.write, target=TMP_PATH / "trash.wav")
        sys.exit(0)

    if args.read:
        # consume audio file
        ds = BeatDataset(BEAT_PATH, args.slice_length) # create just to read labels in use
        labels = ds.labels
        num_classes = len(labels)
        stepsize = args.slice_length // 5
        batch_size = 64
        stride = args.slice_length // stepsize
        ds = RecordedDataset(TMP_PATH / "trash.wav", args.slice_length, stepsize=stepsize)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=1)
        model = SymbolDetector.load_from_checkpoint(ckpt_best)
        model = model.eval()
        results = []
        starttime = time.time()
        results = torch.empty(len(ds), num_classes, dtype=torch.float32)
        idx = 0
        for bdx, batch in enumerate(loader):
            # batch consists of just x since we don't know y
            logits = model(batch)
            smlogits = torch.nn.functional.softmax(logits, dim=1)
            results[idx : idx + smlogits.size(0)] = smlogits
            idx += smlogits.size(0)
        log.info(f"processed in {time.time()-starttime:.2f} s")
        decoded = ""
        for idx, cdx in enumerate(results.argmax(dim=1)):
            if idx % stride == 0:
                decoded += labels[cdx]
        log.info(f"Decoded as {decoded}")
        plt.figure(figsize=(10, 7))
        xxx = np.linspace(0, results.size(0) / (stride), len(ds.samples))
        plt.plot(xxx, ds.samples / 2 - 1, lw=0.3)
        for ddx in range(results.size(0) // (stride)):
            # print(decoded[ddx*(slice_length//stepsize)])
            plt.text(ddx, 0.5, decoded[ddx])
        xxx = np.linspace(0, results.size(0) / (args.slice_length // stepsize), results.size(0))
        for cdx in range(len(labels)):
            char = labels[cdx]
            # if char in "abcd":
            plt.plot(xxx, results[:, cdx].detach().numpy(), label=char)
            # plt.axvline(44.1/2 + 44100//5//2/100 * cdx)

        plt.show()
        sys.exit(0)

    if args.benchmark:
        # benchmark dataloader
        dm = BeatDatamodule(slice_length=args.slice_length)
        dm.setup()
        loader = dm.train_dataloader()
        for _ in range(5):
            starttime = time.time()
            for batch in loader:
                pass
            log.info(f"read training epoch in {time.time() - starttime:.3f} s")
        sys.exit(0)

    if args.train:
        # do full curriculum training
        # allow 30 epochs per lesson, infinite in final
        ds = BeatDataset(BEAT_PATH, args.slice_length, "test")
        num_classes = ds.num_classes

        log.info("lesson 0: easy")
        dm = BeatDatamodule(slice_length=args.slice_length, add_noise=False)
        model = SymbolDetector(num_classes)
        # model = SymbolDetector.load_from_checkpoint(ckpt_best)
        # sample_data = ds[0][0].unsqueeze(0)
        # _ = summary(model, input_data=sample_data)
        # model.freeze_submodel(requires_grad=False)
        trainer = L.Trainer(max_epochs=30, **create_callbacks_loggers(0))
        trainer.fit(model, datamodule=dm)
        ckpt_best = trainer.checkpoint_callback.best_model_path
        log.info(f"lesson 0: finished with {ckpt_best}")

        log.info("lesson 1: medium (add noise)")
        dm.data_train.add_noise = True
        dm.data_val.add_noise = True
        model = SymbolDetector.load_from_checkpoint(ckpt_best)
        # model.freeze_submodel(requires_grad=False)
        trainer = L.Trainer(max_epochs=30, **create_callbacks_loggers(1))
        trainer.fit(model, datamodule=dm)
        ckpt_best = trainer.checkpoint_callback.best_model_path
        log.info(f"lesson 1: finished with {ckpt_best}")

        log.info("lesson 2: hard (low SNR)")
        dm.data_val.val_test_snr_db = 0
        dm.data_train.set_hard_train_snr_db()
        model = SymbolDetector.load_from_checkpoint(ckpt_best)
        trainer = L.Trainer(max_epochs=-1, **create_callbacks_loggers(2))
        trainer.fit(model, datamodule=dm)
        ckpt_best = trainer.checkpoint_callback.best_model_path
        log.info(f"lesson 2: finished with {ckpt_best}")

    if args.eval:
        log.info("evaluation start")
        model = SymbolDetector.load_from_checkpoint(ckpt_best)
        model = model.eval()
        ds = BeatDataset(BEAT_PATH, args.slice_length, "val")
        num_classes = ds.num_classes
        loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=2)
        for snr_db in np.linspace(10, -10, 11):
            ds.val_test_snr_db = snr_db
            confusion = MulticlassConfusionMatrix(num_classes=num_classes)
            accuracy = MulticlassAccuracy(num_classes=num_classes, average=None)
            for batch in loader:
                xxx, yyy = batch
                logits = model(xxx)
                confusion.update(preds=logits, target=yyy)
                accuracy.update(preds=logits, target=yyy)
            if snr_db in [0, 10]:
                confusion.plot(labels=ds.labels)
                accuracy.plot()
                plt.show()
            # print(accuracy.compute())
            log.info(f"{ds.val_test_snr_db:+5.1f} dB SNR Accuracy = {accuracy.compute().mean():.2%}")
