from typing import List
from xml.parsers.expat import model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torchaudio, torch.nn as nn, torchvision, torchvision.models as models, numpy as np, torch, gc, torch.nn.functional as F, torch.optim as optim, warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from PIL import Image
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

####################################################

# pre processing

def normalize_per_sensor(waveforms):
    N, T, F = waveforms.shape
    out = waveforms.copy().astype(np.float32)
    for f in range(F):
        mu  = out[:, :, f].mean()
        std = out[:, :, f].std()
        out[:, :, f] = (out[:, :, f] - mu) / std
    return out

def signal_to_mel_img(data, device):
    
    # SAMPLE_RATE = 2_500_000  # Hz (2.5 MHz) - from 400 ns sampling period: 1/(400e-9) = 2.5e6
    # N_FFT = 256              # FFT window size (smaller = better time resolution)
    # HOP_LENGTH = 16          # Overlap (smaller = more frames, less info loss)
    # N_MELS = 128        # Mel frequency bins (more = finer frequency detail)
    # F_MIN = 0           # Min frequency
    # F_MAX = SAMPLE_RATE // 2  # Nyquist frequency

    wav2mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=240,
        n_fft=32,
        # win_length=N_FFT,
        hop_length=4,
        # f_min=F_MIN,
        # f_max=F_MAX,
        n_mels=32,
        power=2.0,           # Power spectrogram
    ).to(device)

    amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

    resize2img = torchvision.transforms.Resize((224, 224), antialias=True).to(device)

    wav = torch.from_numpy(data).float().unsqueeze(0).to(device)
    mel = wav2mel(wav)
    mel = amp2db(mel)

    # Min-Max normalize to [0,1]
    vmin, vmax = mel.min(), mel.max()
    mel = (mel - vmin) / (vmax - vmin) if vmax > vmin else torch.zeros_like(mel)

    img = mel.repeat(3, 1, 1)
    img = resize2img(img)
    torch.cuda.empty_cache()
    return img

def build_image_tensor(data, device):
    N = data.shape[0]
    N_SENSORS = data.shape[2]
    all_img = []

    for i in range(N):
        sensor_imgs = [signal_to_mel_img(data[i, :, s], device) for s in range(N_SENSORS)]
        all_img.append(torch.stack(sensor_imgs))
    torch.cuda.empty_cache()
    result = torch.stack(all_img)
    torch.cuda.empty_cache()
    return result

####################################################

# VGG16 pretrianed

class VGG16_FC1(nn.Module):
 
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            vgg.features,            # conv blocks
            vgg.avgpool,             # adaptive avg pool → (512, 7, 7)
            nn.Flatten(),            # → 25088
            vgg.classifier[0],       # Linear(25088, 4096)
            vgg.classifier[1],       # ReLU()
        )
        for p in self.parameters():
            p.requires_grad = False
 
    def forward(self, x: torch.Tensor):
        return self.backbone(x)      # (B, 4096)


def extract_features(images, device, batch_size=32):
    
    model = VGG16_FC1().eval().to(device)
    N = images.shape[0]
    N_SENSORS = images.shape[1]
    IMG_SIZE = images.shape[3]
    all_feats = []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = images[i : i + batch_size]              # (B, 14, 3, 224, 224)
            B     = batch.shape[0]
            flat  = batch.view(B * N_SENSORS, 3, IMG_SIZE, IMG_SIZE).to(device)
            feats = model(flat)                          # (B*14, 4096)
            fused = feats.view(B, N_SENSORS * 4096).cpu()   # (B, 57344)
            all_feats.append(fused)
 
    result = torch.cat(all_feats, dim=0)   # (N, 57344)

    return result


def split_and_shuffle(img_feat, label, train_norm_range, test_norm_range, test_fault_range):
    
    img_feat_norm = img_feat[label==0]
    img_feat_fault = img_feat[label==1]

    train_norm_ind = torch.randperm(train_norm_range)
    test_norm_ind = torch.randperm(test_norm_range)
    test_fault_ind = torch.randperm(test_fault_range)

    train = img_feat_norm[train_norm_ind]

    test_norm = img_feat_norm[test_norm_ind]
    test_fault = img_feat_fault[test_fault_ind]

    # Create labels for test set
    test_norm_labels = torch.zeros(len(test_norm), dtype=torch.int)
    test_fault_labels = torch.ones(len(test_fault), dtype=torch.int)

    # Combine
    test = torch.concat((test_norm, test_fault))
    test_labels = torch.concat((test_norm_labels, test_fault_labels))

    # Shuffle together
    mix_ind = torch.randperm(len(test))
    test = test[mix_ind]
    test_labels = test_labels[mix_ind].cpu().numpy()

    return train, test, test_labels

def split_and_shuffle_phase2(img_feat, label, n_train, test_norm_range, test_fault_range, n_sensors=14, feat_dim=4096):

    img_feat_norm  = img_feat[label == 0]
    img_feat_fault = img_feat[label == 1]

    # base training split
    train_base = img_feat_norm[torch.randperm(n_train)]

    # generate one sensor-dropped copy per sensor → 14× the data
    dropped_copies = []
    for s in range(n_sensors):
        copy = train_base.clone()
        copy[:, s*feat_dim : (s+1)*feat_dim] = 0.0
        dropped_copies.append(copy)

    train_augmented = torch.cat([train_base] + dropped_copies, dim=0)
    # shape: (n_train * 15, 57344)

    # test set — same as Phase 1
    test_norm  = img_feat_norm[torch.randperm(test_norm_range)]
    test_fault = img_feat_fault[torch.randperm(test_fault_range)]
    test_norm_labels  = torch.zeros(len(test_norm),  dtype=torch.int)
    test_fault_labels = torch.ones(len(test_fault), dtype=torch.int)

    test        = torch.cat([test_norm, test_fault])
    test_labels = torch.cat([test_norm_labels, test_fault_labels])
    mix_ind     = torch.randperm(len(test))

    return train_augmented, test[mix_ind], test_labels[mix_ind].cpu().numpy()

####################################################

# Autoencoder model

class Autoencoder(nn.Module):

    def __init__(self, input_dim = 14 *4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(), 
            nn.Linear(4096, 2048),
            nn.ReLU(), 
            nn.Linear(2048, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Linear(512, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 2048),
            nn.ReLU(), 
            nn.Linear(2048, 4096),
            nn.ReLU(), 
            nn.Linear(4096, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoencoderPhase2(nn.Module):
    def __init__(self, input_dim=14*4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),        # helps with zeroed sensor blocks
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class FeatureNormalizer:

    def __init__(self):
        self.vmin = None
        self.vmax = None

    def fit(self, features):
        self.vmin = features.min(dim=0, keepdim=True).values
        self.vmax = features.max(dim=0, keepdim=True).values
        return self

    def transform(self, features):
        x = (features - self.vmin) / (self.vmax - self.vmin + 1e-8)
        return torch.clamp(x, 0.0, 1.0)

    def fit_transform(self, features):
        return self.fit(features).transform(features)
    

####################################################

# training loop

def train_autoencoder(model, features_data, normalizer, n_epochs, lr, device):
    train_norm = normalizer.fit_transform(features_data)
    dataset = TensorDataset(train_norm)
    loader     = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion  = nn.MSELoss()
    history    = []

    print(f"\n  Training AE  ({len(train_norm)} samples, {n_epochs} epochs)")
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        
        epoch_loss /= len(train_norm)
        history.append(epoch_loss)
        scheduler.step()

        print(f"Epoch {epoch+1:>4}/{n_epochs}  Loss: {epoch_loss:.6f}")
        
    return history, normalizer


@torch.no_grad()
def reconstruction_error(model, features_raw, normalizer, device,
                         BATCH_SIZE=32, N_SENSORS=14, FEAT_DIM=4096,
                         agg="max"):
    """
    agg: how to collapse 14 sensor errors into one score per sample
         "max"  — flags a sample if ANY sensor reconstructs poorly
         "mean" — original behaviour, but computed sensor-wise
         "none" — returns (N, 14) array, one error per sensor per sample
    """
    model.eval().to(device)
    feats_norm = normalizer.transform(features_raw)
    errors = []

    for i in range(0, len(feats_norm), BATCH_SIZE):
        batch = feats_norm[i : i + BATCH_SIZE].to(device)  # (B, 57344)
        recon = model(batch)                                # (B, 57344)

        diff = (batch - recon) ** 2                        # (B, 57344)

        # reshape into (B, 14, 4096) — one block per sensor
        diff = diff.view(-1, N_SENSORS, FEAT_DIM)          # (B, 14, 4096)

        # MSE within each sensor block → (B, 14)
        sensor_errors = diff.mean(dim=2)
        sensor_errors = torch.clamp(sensor_errors, max=10)

        errors.append(sensor_errors.cpu().numpy())

    sensor_errors = np.concatenate(errors, axis=0)         # (N, 14)

    if agg == "max":
        return sensor_errors.max(axis=1)                   # (N,)
    elif agg == "mean":
        return sensor_errors.mean(axis=1)                  # (N,)
    elif agg == "none":
        return sensor_errors                               # (N, 14)
    else:
        raise ValueError(f"Unknown agg: {agg}")

@torch.no_grad()
def evaluate_sensor_loss(model, img_feat_norm, img_feat_fault,
                         normalizer, device,percentile,
                         n_test_norm=190, n_test_fault=10,
                         n_sensors=14, feat_dim=4096):
    """
    Evaluates model on 14 test sets, each with one sensor zeroed out.
    Returns a dict of metrics per dropped sensor.
    """
    sensor_names = [
        "A+IGBT-I", "A+*IGBT-I", "B+IGBT-I", "B+*IGBT-I",
        "C+IGBT-I", "C+*IGBT-I", "A-Flux", "B-Flux", "C-Flux",
        "Mod-V", "Mod-I", "CB-I", "CB-V", "DV/DT"
    ]

    results = {}

    for s in range(n_sensors):
        # build test set with sensor s dropped
        test_norm  = img_feat_norm[torch.randperm(n_test_norm)]
        test_fault = img_feat_fault[torch.randperm(n_test_fault)]

        test_norm[:,  s*feat_dim : (s+1)*feat_dim] = 0.0
        test_fault[:, s*feat_dim : (s+1)*feat_dim] = 0.0

        test        = torch.cat([test_norm, test_fault])
        test_labels = np.array([0]*n_test_norm + [1]*n_test_fault)

        mix_ind     = torch.randperm(len(test))
        test        = test[mix_ind]
        test_labels = test_labels[mix_ind]

        errors = reconstruction_error(model=model, features_raw=test,
                                      normalizer=normalizer, device=device,
                                      agg="max")

        thresh = np.percentile(
            reconstruction_error(model=model, features_raw=img_feat_norm,
                                 normalizer=normalizer, device=device, agg="max"),
            percentile
        )

        metrics = compute_metrics(errors, test_labels, thresh,
                                  label=f"drop {sensor_names[s]}")
        results[sensor_names[s]] = metrics

    return results

def threshold_by_f1(train_error, test_error, test_labels):
    best_thresh, best_f1, best_p = 0, 0, 0
    for p in range(80, 100):
        t = np.percentile(train_error, p)
        preds = (test_error > t).astype(int)
        f1 = f1_score(test_labels, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
            best_p = p

    print(f"Best percentile: {best_p}  threshold: {best_thresh:.4f}  F1: {best_f1:.4f}")
    return best_p, best_thresh

####################################################

# metrics

def compute_metrics(errors,true_labels,threshold,label=""):

    preds = (errors > threshold).astype(int)
 
    eps       = 1e-8
    precision = precision_score(true_labels, preds)
    recall    = recall_score(true_labels, preds)
    accuracy  = accuracy_score(true_labels, preds)
    f1        = f1_score(true_labels, preds)
 
    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, errors)
    else:
        auc = float("nan")
 
    metrics = dict(
        precision=precision, recall=recall, accuracy=accuracy,
        f1=f1, auc=auc,
    )

    print(confusion_matrix(true_labels, preds))
 
    tag = f" [{label}]" if label else ""
    print(f"\n{'─'*52}")
    print(f"  Evaluation{tag}")
    print(f"{'─'*52}")
    for k in ["precision", "recall", "accuracy", "f1", "auc"]:
        print(f"  {k:<12} {metrics[k]:.4f}")

####################################################

# Plot

BLUE  = "#534AB7"
GREEN = "#1D9E75"
CORAL = "#D85A30"
GRAY  = "#888780"
 
 
def plot_training_curve(h, title="Phase 1 — Shallow AE"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(h, color=BLUE, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
 
 
def plot_error_distributions(errors, labels, threshold, title, filename=None):
    norm_err  = errors[labels == 0]
    fault_err = errors[labels == 1]
    fig, ax   = plt.subplots(figsize=(9, 4))
    ax.hist(norm_err,  bins=40, alpha=0.65, color=GREEN, label="Normal",    density=True)
    ax.hist(fault_err, bins=40, alpha=0.65, color=CORAL, label="Anomalous", density=True)
    ax.axvline(threshold, color=BLUE, linestyle="--", linewidth=1.8,
               label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()
