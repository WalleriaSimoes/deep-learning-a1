import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
import cv2
import time


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


import matplotlib.colors as mcolors
from scipy.ndimage import label
from skimage.segmentation import watershed
from torchvision.utils import make_grid


# Variaveis globais
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class BBBC038Dataset(Dataset):

    def __init__(self, root, transform):
        self.root = Path(root)

        # Encontra todas as imagens
        self.images = sorted(self.root.glob("*/images/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        # Pasta correspondente a imagem
        sample_dir = image_path.parent.parent

        # Carrega a imagem em um tensor
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (BGR ==> RGB)

        # Carrega as mascaras
        mask_paths = sorted((sample_dir / "masks").glob("*.png"))

        masks = []

        for i in range(len(mask_paths)):
            mask_path = mask_paths[i]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask != 0) * (i + 1)
            masks.append(mask)

        mask = np.stack(masks).max(0).astype(int)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0).float()

        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(128,256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256,128,kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256,128)
        self.up2 = nn.ConvTranspose2d(128,64,kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128,64)

        # Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Passa pelo primeiro bloco e salva
        skip1 = self.conv1(x)
        x = self.pool1(skip1)
        skip2 = self.conv2(x)
        x = self.pool2(skip2)


        x = self.bottleneck(x)
        x = self.up1(x)
        x = torch.cat((skip2, x), dim=1)
        x = self.dec1(x)


        x = self.up2(x)
        x = torch.cat((skip1, x), dim=1)
        x = self.dec2(x)

        return self.final_conv(x)


def binary_decoder(mask, threshold):
    return (torch.sigmoid(mask) > threshold).float()

def load_best_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    path_weights = path
    weights = torch.load(path_weights, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    return model

def val_loop(model, loader_val, transform_mask, loss, decoder, metric_function,**kwargs):
    model.eval()
    loss_val = 0.0
    total_metrics = {}
    img_count = 0
    with torch.no_grad():
        for img, mask_original in loader_val:
            img = img.to(device)
            mask_original = mask_original.to(device)
            # Aplica uma transformacao na mascara, caso desejado
            if not transform_mask is None:
                mask = transform_mask(mask_original)
            else:
                mask = mask_original

            pred = model.forward(img)
            loss_value = loss(pred, mask)
            loss_val += loss_value.item()
            pred = decoder(pred, **kwargs)
            # print(img.shape)
            for b in range(img.shape[0]):
                # Chama a função de métrica que foi passada como parâmetro
                metrics = metric_function(pred[b], mask_original[b])

                # Acumula as métricas dinamicamente
                for key, val in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + val
                img_count += 1
    mean_loss_val = loss_val / len(loader_val)
    mean_metrics = {k: v / img_count for k, v in total_metrics.items()}

    return mean_loss_val, mean_metrics

def binarize_mask(mask):
    return (mask > 0).float()

def binary_decoder(mask, threshold):
    return (torch.sigmoid(mask) > threshold).float()

def multiclass_decoder(mask, dim):
    return torch.argmax(mask, dim=dim).float()

def border_mask(mask, kernel_size=3):
    padding = kernel_size // 2

    # Obtem o minimo e o maximo na vizinhanca
    mask_float = mask.float()
    max_pool   = F.max_pool2d(mask_float, kernel_size=kernel_size, stride=1, padding=padding)
    min_pool   = -F.max_pool2d(-mask_float, kernel_size=kernel_size, stride=1, padding=padding)

    # Caso o minimo e o maximo sejam diferentes, eh fronteira
    mask_binary = binarize_mask(mask)
    mask_binary += torch.where(mask_binary != 0, (max_pool != min_pool).float(), 0.0)

    return mask_binary.long().squeeze(1) # Fronteira = 2, Interior = 1 e Fundo = 0

def watershed_decoder(mask):
    images = mask.shape[0]
    # Considera somente regioes internas
    interior_mask = (mask == 1)

    # Monta as componentes conexas
    interior_mask = np.stack(
        [
            label(
                interior_mask[i].squeeze(0).cpu().numpy()
            )[0]
            for i in range(images)
        ]
    )

    # Expande (Watershed)
    cell_pixes = (mask != 0).float().cpu().numpy()
    image = np.zeros_like(interior_mask[0])
    return torch.stack(
        [
            torch.tensor(watershed(image=image, markers=interior_mask[i], mask=cell_pixes[i]))
            for i in range(images)
        ]
    ).unsqueeze(1)

def compute_semantic_metrics(pred, mask):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    mask_bin = (mask > 0).float()
    # print(f"Dimensoes do pred e da mask depois do unsqueeze: {pred.shape}, {mask.shape}")
    intersection = torch.sum((pred_bin*mask_bin))
    union = torch.sum((pred_bin+mask_bin)) - intersection
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (torch.sum(pred_bin) + torch.sum(mask_bin) + 1e-8)

    return {'iou': iou.item(), 'dice': dice.item()}


def compute_intersection(mask_true, mask_pred):
    mask_true = np.asarray(mask_true)
    mask_pred = np.asarray(mask_pred)
    
    # Mapeia os ids para numeros sequenciais
    _, mask_true = np.unique(mask_true, return_inverse=True)
    _, mask_pred = np.unique(mask_pred, return_inverse=True)

    num_true_classes = int(np.max(mask_true)) + 1
    num_pred_classes = int(np.max(mask_pred)) + 1

    # Calcula a quantidade de pixels de cada instancia na mascara real
    true_inst_counts = np.unique_counts(mask_true)
    true_instances_counts = dict(zip(true_inst_counts.values, true_inst_counts.counts))

    # Calcula a quantidade de pixels de cada instancia na mascara prevista
    pred_inst_counts = np.unique_counts(mask_pred)
    pred_instances_counts = dict(zip(pred_inst_counts.values, pred_inst_counts.counts))

    # Cria a matriz de interseções com zeros
    intersect_matrix = np.zeros((num_pred_classes, num_true_classes), dtype=float)
    fn_matrix = intersect_matrix.copy()
    fp_matrix = intersect_matrix.copy()
    intersect = np.hstack([mask_pred.reshape(-1, 1), mask_true.reshape(-1, 1)]).astype(int)
    intersect_values, intersect_counts = np.unique(intersect, axis=0, return_counts=True)

    # Soma as intersecoes a matrix
    for (pred_inst, true_inst), qtd in zip(list(intersect_values), intersect_counts):
        # Ignora pixels que nao devem ser comparados (para as proximas questoes)
        if pred_inst == -1 or true_inst == -1: 
            continue
        qtd = float(qtd)
        intersect_matrix[pred_inst][true_inst] += qtd
        fn_matrix[pred_inst][true_inst] = true_instances_counts.get(true_inst, 0) - qtd
        fp_matrix[pred_inst][true_inst] = pred_instances_counts.get(pred_inst, 0) - qtd


    # Computa o iou
    iou_matrix = intersect_matrix.copy()
    iou_matrix[iou_matrix != 0] = iou_matrix[iou_matrix != 0] / (
        iou_matrix[iou_matrix != 0]
        + fn_matrix[iou_matrix != 0]
        + fp_matrix[iou_matrix != 0]
    )

    return intersect_matrix, iou_matrix

def greedy_match(iou, predicts, trues):
    iou = iou.copy()
    # Faz o casamento de forma gulosa
    matches_iou = np.full(shape=predicts, fill_value=0.0)
    matches = np.full(shape=predicts, fill_value=None)
    for _ in range(min(predicts, trues)):
        pred_mxs = iou.max(axis=1)
        greedy_pred_isnt = pred_mxs.argmax()
        greedy_true_isnt = iou[greedy_pred_isnt].argmax()
        greedy_iou = iou[greedy_pred_isnt, greedy_true_isnt].max()

        # Apaga virtualmente as instancias prevista e verdadeira
        iou[greedy_pred_isnt, :] = -1
        iou[:, greedy_true_isnt] = -1

        matches_iou[greedy_pred_isnt] = greedy_iou
        matches[greedy_pred_isnt]     = greedy_true_isnt

    return matches_iou, matches

def test_limiares(iou, limiares, predicts, trues):
    matches_iou, _ = greedy_match(iou, predicts, trues)

    true_positives = []
    false_positves = []
    false_negatives = []
    average_precisions = []

    # Calcula TP, FP e FN para cada limiar
    for limiar in limiares:
        tp = (matches_iou >= limiar).sum()
        fp = predicts - tp
        fn = trues - tp
        true_positives.append(tp)
        false_positves.append(fp)
        false_negatives.append(fn)
        average_precisions.append(tp / (tp + fp + fn))

    return true_positives, false_positves, false_negatives, average_precisions

def compute_metrics(iou_matrix, limiares):
    iou = iou_matrix[1:, 1:]
    predicts, trues = iou.shape
    true_positives, false_positves, false_negatives, average_precisions = test_limiares(iou, limiares, predicts, trues)
    mean_average_precision = np.mean(average_precisions)
    abs_count_error = np.abs(predicts - trues)

    return true_positives, false_positves, false_negatives, float(mean_average_precision), float(abs_count_error)

def compute_instance_metrics_wrapper(pred_inst, mask_true_inst):
    pred_np = pred_inst.cpu().numpy()
    mask_true_np = mask_true_inst.cpu().numpy()

    intersect_matrix, iou_matrix = compute_intersection(mask_true_np, pred_np)

    if iou_matrix.shape == (1, 1): # Se não previu nada, o mAP é 0
        return {'map': 0.0, 'abs_err': 0.0}

    _, _, _, mean_ap, abs_err = compute_metrics(iou_matrix, np.arange(0.5, 1, 0.05))

    return {'map': mean_ap, 'abs_err': abs_err}

train_transform = A.Compose([
    # Forca image e mask para 256x256
    A.Resize(height=256, width=256),

    # Padroniza o espectro de cores como cinza
    A.ToGray(p=1.0),

    # Normaliza os pixels para o padrao ImageNet
    A.Normalize(mean=MEAN, std=STD),

    # Converte os arrays Numpy para Tensores do PyTorch e ajusta as dimensoes ((H, W, C) ==> (C, H, W))
    ToTensorV2()
])

batch_size = 16
dataset = BBBC038Dataset("data/stage1_train", train_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size)
train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [0.8,0.2])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_best_model('models/melhor_modelo.pth')

model.eval()

total_map = 0.0
total_abs_err = 0.0
img_count = 0

print("Iniciando avaliação do conjunto de validação...")
start_time = time.time()

with torch.no_grad():
    for img, mask_original_batch in val_dataloader:
        img = img.to(device)
        mask_original_batch = mask_original_batch.to(device)

        # Inferência
        pred_logits = model(img) 
        pred_classes = multiclass_decoder(pred_logits, dim=1).cpu().numpy().astype(int)

        # Pós-processamento (Watershed)
        mask_watershed_batch = watershed_decoder(torch.tensor(pred_classes))

        # Cálculo de Métricas por imagem
        for b in range(img.shape[0]):
            mask_orig = mask_original_batch[b, 0]
            mask_water = mask_watershed_batch[b, 0]

            metrics = compute_instance_metrics_wrapper(mask_water, mask_orig)
            
            total_map += metrics['map']
            total_abs_err += metrics['abs_err']
            img_count += 1

# Cálculo das médias finais
mean_map = total_map / img_count
mean_abs_err = total_abs_err / img_count
end_time = time.time()

print("-" * 30)
print("RESULTADOS DA AVALIAÇÃO")
print("-" * 30)
print(f"Imagens Avaliadas: {img_count}")
print(f"mAP Médio:         {mean_map:.4f}")
print(f"Erro Abs. Médio:   {mean_abs_err:.4f}")
print(f"Tempo de Execução: {end_time - start_time:.2f} segundos")