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


#########################################################################
#                           VARIAVEIS GLOBAIS                           #
#########################################################################
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
BATCH_SIZE = 16


#########################################################################
#                          FUNCOES AUXILIARES                           #
#########################################################################
def binarize_mask(mask):
    return (mask > 0).float()

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

def tiles_mask(mask, tamanho_janela=256, stride=128, mk_grid=True):
    # Junta as images do batch num unico grid, caso ela ja nao venha como uma imagem sobig_mask = mask.clone()
    if mk_grid:
        c = mask.shape[1]
        big_mask = make_grid(mask, nrow=4, padding=0)[:c, :, :]
    else:
        big_mask = mask.squeeze(0).clone()
    
    # Obtem os recortes com stride e tamanho de janela definidos, empurrando as dimensoes extras para o tamanho do novo batch
    return torch.cat(
        [
            channel
            .unfold(0, tamanho_janela, stride)
            .unfold(1, tamanho_janela, stride)
            .contiguous()
            .view(-1, 1, tamanho_janela, tamanho_janela)
            for channel in big_mask
        ],
        dim=1
    )
    
def tiles_unmask(mask, tamanho_imagem=[1024, 1024], tamanho_janela=256, stride=128):
    unmask = mask.contiguous().view(mask.size(0), -1).transpose(0, 1).unsqueeze(0)
    
    # Desfaz o Unfold, somando as fronteiras compartilhadas
    unmask_sum = F.fold(
        unmask, 
        output_size=tamanho_imagem, 
        kernel_size=tamanho_janela, 
        stride=stride
    )
    
    # Faz a contagem de intersecoes
    ones_flat = torch.ones_like(unmask)
    count = F.fold(
        ones_flat, 
        output_size=tamanho_imagem, 
        kernel_size=tamanho_janela, 
        stride=stride
    )
    
    # Dividir para obter a media suave nas bordas
    return unmask_sum / count

def tiles_naive_unmask(mask, tamanho_imagem=[1024, 1024], tamanho_janela=256, stride=128):
    tamanho_imagem = np.asarray(tamanho_imagem).astype(int)
    mod = ((tamanho_imagem / stride) - 1).astype(int)
    
    naive_mask = torch.zeros(size=tuple(tamanho_imagem))
    
    start_y = 0
    end_y   = tamanho_janela
    current = 0
    for each_row in range(mod[0]):
        start_mask_row = (tamanho_janela - stride) * (each_row != 0) 
        start_x        = 0
        end_x          = tamanho_janela
        for each_col in range(mod[1]):
            start_mask_col = (tamanho_janela - stride) * (each_col != 0) 
                
            naive_mask[start_y:end_y, start_x:end_x] += mask[current, 0, start_mask_row:, start_mask_col:]
            
            start_x  = end_x
            end_x   += stride
            current += 1
                
        start_y = end_y
        end_y  += stride
        
    return naive_mask.unsqueeze(0).unsqueeze(1)

def binary_decoder(mask, threshold):
    return (torch.sigmoid(mask) > threshold).float()

def multiclass_decoder(mask, dim):
    return torch.argmax(mask, dim=dim).float()

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


def multiclass_watershed_decoder(pred_logits, **kwargs):
    pred_classes = torch.argmax(pred_logits, dim=1)
    return watershed_decoder(pred_classes)

def compute_instance_metrics_wrapper(pred_inst, mask_true_inst):
    pred_np = pred_inst.cpu().numpy()
    mask_true_np = mask_true_inst.cpu().numpy()

    intersect_matrix, iou_matrix = compute_intersection(mask_true_np, pred_np)

    if iou_matrix.shape == (1, 1): # Se não previu nada, o mAP é 0
        return {'map': 0.0, 'abs_err': 0.0}

    _, _, _, mean_ap, abs_err = compute_metrics(iou_matrix, np.arange(0.5, 1, 0.05))

    return {'map': mean_ap, 'abs_err': abs_err}


def desnorm(img, mean=MEAN, std=STD):
    img_plot = img.cpu().permute(1, 2, 0).numpy()
    mean = np.array(mean)
    std = np.array(std)
    img_plot = (img_plot * std) + mean
    return np.clip(img_plot, 0, 1)


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


def get_weights(loader, classes, transform_mask=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_counts = torch.zeros(classes, dtype=torch.float64).to(device)
    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            mask = mask.to(device)
            if not transform_mask is None:
                mask = transform_mask(mask)

            # Estava dando problema caso não tivesse uma das classes, coloquei um bincount pra contornar isso
            counts = torch.bincount(mask.flatten(), minlength=classes)
            total_counts += counts

    mx = total_counts.max()
    weights = mx / (total_counts + 1e-8)

    return weights.float()

def load_best_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    path_weights = path
    weights = torch.load(path_weights, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    return model

def map_to_seq(mask):
    uniques   = torch.unique(mask[mask != -1]).cpu().numpy()
    idxs      = np.arange(uniques.size) + (0 not in uniques)
    map_front = {u: i for u, i in zip(uniques, idxs)}
    map_inv   = {i: u for u, i in zip(uniques, idxs)}
    return map_front, map_inv
    

def fusion(intersect_ant, intersect_pos, next_class):
    intersect_ant = intersect_ant.clone()
    intersect_pos = intersect_pos.clone()
    
    # Faz um map das classes para numeros sequencias
    map_ant_front, map_ant_inv = map_to_seq(intersect_ant)
    map_pos_front, map_pos_inv = map_to_seq(intersect_pos)
    
    # Calcula a matriz de iou e os matches
    _, iou_matrix = compute_intersection(mask_true=intersect_ant, mask_pred=intersect_pos)

    iou = iou_matrix[1:, 1:]
    predicts, trues = iou.shape
    matches_iou, matches = greedy_match(iou, predicts, trues)
    
    # Retorna o match com as classes originais
    matches_fusion = {}
    for pos, ant in enumerate(matches):
        pos = -1 if pos is None else map_pos_inv[pos+1]
        ant = -1 if ant is None else map_ant_inv[ant+1]
        if ant == -1:
            matches_fusion[pos] = next_class
            next_class += 1
        else:
            matches_fusion[pos] = ant
    
    return matches_fusion, next_class
    
    
    
def inst_fusion(pred, tamanho_imagem=[1024, 1024], stride=128):
    pred = pred.clone()
    height, width = pred.shape[-2:]
    
    # Calcula a quantidade de tiles por eixo
    tamanho_imagem = np.asarray(tamanho_imagem).astype(int)
    mod            = ((tamanho_imagem / stride) - 1).astype(int)
    
    current    = 0
    next_class = 0
    for each_row in range(mod[0]):
        # Em cada recorte, avalia as intersecoes anteriores (a esquerda, em cima e a em cima/esquerda)
        for each_col in range(mod[1]):
            if not any([each_col, each_row]):
                current   += 1
                next_class = pred[0, 0].max() + 1
                continue
            
            # Calcula os indices dos sucessoros
            sup_esq = sup = esq = None
            if each_row:
                sup     = (current - mod[1])
                sup_esq = (sup - 1) if each_col else None
            if each_col:
                esq = (current - 1)
            
            # Constroi uma mask com os ids ja selecionados
            ant_pred     = torch.full_like(pred[0], fill_value=0)
            pred_current = torch.full_like(pred[0], fill_value=0)
            for ant_idx, (is_ant_esq, is_ant_sup) in zip(
                [sup_esq, sup, esq],
                [(True, True), (False, True), (True, False)]
            ):
                if ant_idx is None: continue
                stride_x = (stride * int(is_ant_esq))
                stride_y = (stride * int(is_ant_sup))
                ant_pred[0, :(height-stride_y), :(width-stride_x)]     = pred[ant_idx, 0, stride_y:, stride_x:]
                pred_current[0, :(height-stride_y), :(width-stride_x)] = pred[current, 0, :(height-stride_y), :(width-stride_x)]
            
            # Processa as intersecoes com a regiao ja selecionada, trocando os ids de acordo com os matchings
            matches_fusion, next_class = fusion(ant_pred, pred_current, next_class)
            for each_pred, each_ant in matches_fusion.items():
                pred[current, 0] = torch.where(pred[current, 0] == each_pred, each_ant, pred[current, 0])
            
            current += 1
        
    return pred

def globalize_ids(mask):
    mxs = mask.flatten(1).max(dim=1).values
    acc_sum = torch.cat([torch.zeros(1, dtype=mxs.dtype, device=mask.device), torch.cumsum(mxs, dim=0)[:-1]])
    nao_fundo = (mask != 0)
    return torch.where(nao_fundo, mask + acc_sum.view(-1, 1, 1, 1), mask)


# Primeiro, vamos calcular a distribuição de tamanhos dos objetos do dataset e o campo receptivo teórico do encoder
# Para o campo receptivo do encoder, usamos como base o pseudocódigo do link: https://www.baeldung.com/cs/cnn-receptive-field-size (que eu descobri depois que estava errado ;-;, mas já corrigi!)

def compute_receptive_field(k, s, L=2):
    r = 1

    for l in range(1,L+1):
        S = 1

        for i in range(1,l):
            S *= s[i-1]

        r += (k[l-1]-1)*S

    return r


def compute_objects_size_distribution(dataloader):
    sizes = []

    max_size = 0
    max_img = None
    max_mask = None

    min_size = float('inf')
    min_img = None
    min_mask = None

    for img, mask_original in dataloader:
        for i in range(mask_original.shape[0]):
            mask = mask_original[i].squeeze(0).cpu().numpy()
            classes = np.unique(mask)

            for each_class in classes:
                if not each_class:
                    continue

                xs, ys = np.where(mask == each_class)
                height = ys.max() - ys.min() + 1
                width = xs.max() - xs.min() + 1

                current_size = max(height, width)
                sizes.append(current_size)

                if current_size > max_size:
                    max_size = current_size
                    max_img = img[i]
                    max_mask = mask

                if current_size < min_size:
                    min_size = current_size
                    min_img = img[i]
                    min_mask = mask

    return sizes, max_size, max_img, max_mask, min_size, min_img, min_mask





#########################################################################
#                       DEFINICAO DE DATA LOADERS                       #
#########################################################################
def generateEllipses(img_size):
    # Vamos adicinar um contraste para o fundo da imagem
    bg_intensity = np.random.randint(20,80)
    img = np.full((img_size, img_size), bg_intensity, dtype=np.uint8)

    # Evitar overflow
    mask = np.zeros((img_size, img_size), dtype=np.int32)

    num_ellipses = np.random.randint(15,21)

    # Para cada elipse, define o formato, a posição e a intensidade
    for i in range(1, num_ellipses+1):
        center = (int(np.random.randint(15, img_size-15)), int(np.random.randint(15, img_size-15)))
        axes = (int(np.random.randint(5,25)), int(np.random.randint(5,25)))
        angle = int(np.random.randint(0,180))
        intensity = int(np.random.randint(100,255))

        cv2.ellipse(img, center, axes, angle, 0, 360, intensity, -1)
        cv2.ellipse(mask, center, axes, angle, 0, 360, i, -1)

    std_dev = np.random.uniform(5.0, 30.0)
    noise = np.random.uniform(0, std_dev, (img_size, img_size))

    noisy_img = np.clip(img + noise, 0, 255)

    return noisy_img, mask


class EllipsesDataset(Dataset):
  def __init__(self, num_samples) -> None:
     self.num_samples = num_samples

  def __len__(self):
    return self.num_samples

  def __getitem__(self, index):
    img, mask = generateEllipses(128)
    tensor_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    tensor_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    return tensor_img, tensor_mask


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


def corrupt_transform(intensity):
    if intensity == 0:
        brightness_limit = 0.1
        contrast_limit   = 0.1
        std_range        = (0.05, 0.15)
        blur_limit       = (3, 5)
    elif intensity == 1:
        brightness_limit = 0.25
        contrast_limit   = 0.25
        std_range        = (0.15, 0.30)
        blur_limit       = (5, 9)
    else:
        brightness_limit = 0.5
        contrast_limit   = 0.5
        std_range        = (0.30, 0.5)
        blur_limit       = (9, 15)
        
    return A.Compose([
        # Forca image e mask para 256x256
        A.Resize(height=256, width=256),
        
        # Brilho e Contraste
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit, # Varia o brilho em +/- 20%
            contrast_limit=contrast_limit,   # Varia o contraste em +/- 20%
            p=1.0
        ),

        # Ruido (Simula artefatos de sensores de microscopio ISO alto)
        A.GaussNoise(
            std_range=std_range, # Intensidade do granulado
            p=1.0
        ),

        # Blur (Simula perda de foco da lente ou movimento)
        A.OneOf([
            A.GaussianBlur(blur_limit=blur_limit, p=1.0), # Desfoque suave gaussiano
            A.MedianBlur(blur_limit=blur_limit, p=1.0),        # Desfoque que borra mantendo bordas pesadas
            A.MotionBlur(blur_limit=blur_limit, p=1.0),        # Simula a lamina escorregando
        ], p=1.0), 
        
        # Padroniza o espectro de cores como cinza
        A.ToGray(p=1.0),

        # Normaliza os pixels para o padrao ImageNet
        A.Normalize(mean=MEAN, std=STD),

        # Converte os arrays Numpy para Tensores do PyTorch e ajusta as dimensoes ((H, W, C) ==> (C, H, W))
        ToTensorV2()
    ])



#########################################################################
#                         DEFINICAO DES MODELOS                         #
#########################################################################
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
    
class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()
        # Encoder (Igual ao da UNet)
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck
        self.bottleneck = DoubleConv(128,256)

        # Decoder
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256,128)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128,64)

        # Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x, idx1 = self.pool1(x)
        x = self.conv2(x)
        x, idx2 = self.pool2(x)

        x = self.bottleneck(x)

        x = self.dec1(x)
        x = self.unpool1(x,idx2)
        x = self.dec2(x)
        x = self.unpool2(x,idx1)

        return self.final_conv(x)
    

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()

        # O número de canais de cada nível da pirâmide (dividimos igualmente)
        out_channels_per_pool = in_channels // len(pool_sizes)

        # Criamos as camadas de pooling e redução de canais para cada escala
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, out_channels_per_pool, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels_per_pool),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])

        # No final, concatenamos a entrada original (in_channels) com as 4 saídas da pirâmide
        # E usamos uma convolução para ajustar para o out_channels esperado pelo Decoder da U-Net
        concat_channels = in_channels + (out_channels_per_pool * len(pool_sizes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # A lista começa com o próprio mapa de características original (Alta Resolução)
        out = [x]

        # Passamos a imagem por cada nível da pirâmide (Contexto Global/Intermediário)
        for stage in self.stages:
            pooled = stage(x)
            # Esticamos o resumo de volta para o tamanho do mapa original
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            out.append(upsampled)

        # Juntamos o mapa original com todos os resumos macroscópicos
        out = torch.cat(out, dim=1)

        return self.bottleneck(out)

class ModUNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ModUNet, self).__init__()
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = PyramidPoolingModule(in_channels=128, out_channels=256)

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_semantic_metrics(pred, mask):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    mask_bin = (mask > 0).float()
    # print(f"Dimensoes do pred e da mask depois do unsqueeze: {pred.shape}, {mask.shape}")
    intersection = torch.sum((pred_bin*mask_bin))
    union = torch.sum((pred_bin+mask_bin)) - intersection
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (torch.sum(pred_bin) + torch.sum(mask_bin) + 1e-8)

    return {'iou': iou.item(), 'dice': dice.item()}

def train_loop(model, optimizer, loader_train, transform_mask, loss):
    loss_train = 0.0
    model.train()
    for img, mask in loader_train:
        img = img.to(device)
        mask = mask.to(device)

        # Aplica uma transformacao na mascara, caso desejado
        if not transform_mask is None:
            mask = transform_mask(mask)

        optimizer.zero_grad()
        pred = model.forward(img)
        loss_value = loss(pred, mask)
        loss_train += loss_value.item()
        loss_value.backward()
        optimizer.step()
    mean_loss_train = loss_train / len(loader_train)
    return mean_loss_train

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

def train_model(
    model,
    loader_train,
    loader_val,
    optimizer,
    epochs=50,
    loss=nn.BCEWithLogitsLoss(),
    transform_mask=None,
    decoder=binary_decoder,
    metric_function = compute_semantic_metrics,
    stop_metric='iou',
    stop_threshold=0.99,
    patience=5,
    best_model_path='melhor_modelo.pth',
    **kwargs
):

    time_begin = time.time()
    metrics_history = []
    best_metric = -float('inf')
    for epoch in range(epochs):
        mean_loss_train = train_loop(model, optimizer, loader_train, transform_mask, loss)
        mean_loss_val, mean_metrics = val_loop(model, loader_val, transform_mask, loss, decoder, metric_function,**kwargs)
        metrics_history.append((mean_loss_train, mean_loss_val, mean_metrics))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Val Loss: {mean_loss_val:.4f} | Train Loss: {mean_loss_train:.4f}")
            print(" | ".join([f"Val {k.upper()}: {v:.4f}" for k, v in mean_metrics.items()]))

        current_metric = mean_metrics.get(stop_metric, 0.0)

        if current_metric >= stop_threshold:
            print(f"Early stopping")
            break

        if current_metric > best_metric:
            torch.save(model.state_dict(), best_model_path)
            best_metric = current_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: Sem melhoria por {patience} epocas consecutivas.")
                break

    model.load_state_dict(torch.load(best_model_path))
    time_end = time.time()
    print(f"Train time: {(time_end - time_begin)/60} minutes")

    return metrics_history


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


#########################################################################
#                     CARREGAMENTO DOS DATALOADERS                      #
#########################################################################
ellip_data_train = EllipsesDataset(800)
ellip_loader_train = DataLoader(ellip_data_train, batch_size=16)

ellip_data_val = EllipsesDataset(200)
ellip_loader_val = DataLoader(ellip_data_val, batch_size=16)


# Pipeline para redimensionamento das imagens
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

dataset = BBBC038Dataset("data/stage1_train", train_transform)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [0.8,0.2])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

intensity_datasets = {}

for intensity in [0, 1, 2]:
    corrupted_dataset = BBBC038Dataset("data/stage1_train", corrupt_transform(intensity))
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    corrupted_train_dataset, corrupted_val_dataset = torch.utils.data.dataset.random_split(corrupted_dataset, [0.8, 0.2])
    corrupted_train_dataloader = DataLoader(corrupted_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    corrupted_val_dataloader = DataLoader(corrupted_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    intensity_datasets[intensity] = [corrupted_train_dataloader, corrupted_val_dataloader]






#########################################################################
#                           MODELO DA PARTE 0                           #
#########################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_model(
    model,
    loader_train=ellip_loader_train,
    loader_val=ellip_loader_val,
    optimizer=optimizer,
    epochs=50,
    loss=nn.BCEWithLogitsLoss(),
    transform_mask=binarize_mask,
    decoder=binary_decoder,
    metric_function=compute_semantic_metrics,
    stop_metric='iou',
    stop_threshold=0.99,
    patience=5,
    best_model_path="models/best_model_parte_00.pth",
    threshold=0.5
)


#########################################################################
#                           MODELO DA PARTE 1                           #
#########################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_model(
    model,
    loader_train=train_dataloader,
    loader_val=val_dataloader,
    optimizer=optimizer,
    epochs=1,
    loss=nn.BCEWithLogitsLoss(),
    transform_mask=binarize_mask,
    decoder=binary_decoder,
    metric_function=compute_semantic_metrics,
    stop_metric='iou',
    stop_threshold=0.99,
    patience=5, 
    best_model_path="models/best_model_parte_01.pth",
    threshold=0.5
)


#########################################################################
#                           MODELO DA PARTE 2                           #
#########################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
weight = get_weights(loader=train_dataloader, classes=3, transform_mask=border_mask)

train_model(
    model,
    loader_train=train_dataloader,
    loader_val=val_dataloader,
    optimizer=optimizer,
    epochs=30,
    loss=nn.CrossEntropyLoss(weight=weight),
    transform_mask=border_mask,
    decoder=multiclass_watershed_decoder,
    metric_function=compute_instance_metrics_wrapper,
    stop_metric='map',
    stop_threshold=0.99,
    patience=5,
    best_model_path="models/best_model_parte_02.pth",
    dim=1
)

#########################################################################
#                           MODELO DA PARTE 3                           #
#########################################################################
resultados_mAP = [[],[]]
weight = get_weights(loader=train_dataloader, classes=3, transform_mask=border_mask)

for seed in [3,20]:
    set_seed(seed)

    # Treino do modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = train_model(
        model,
        loader_train=train_dataloader,
        loader_val=val_dataloader,
        optimizer=optimizer,
        epochs=1,
        loss=nn.CrossEntropyLoss(weight=weight),
        transform_mask=border_mask,
        decoder=multiclass_watershed_decoder,
        metric_function=compute_instance_metrics_wrapper,
        stop_metric='map',
        stop_threshold=0.99,
        patience=5,
        best_model_path="models/best_model_parte_03_01_UNet.pth",
        dim=1
    )

    best_map = max([epoch_data[2].get('map', 0) for epoch_data in metrics])
    resultados_mAP[0].append(best_map)

    # Treino do modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegNet(in_channels=3, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = train_model(
        model,
        loader_train=train_dataloader,
        loader_val=val_dataloader,
        optimizer=optimizer,
        epochs=1,
        loss=nn.CrossEntropyLoss(weight=weight),
        transform_mask=border_mask,
        decoder=multiclass_watershed_decoder,
        metric_function=compute_instance_metrics_wrapper,
        stop_metric='map',
        stop_threshold=0.99,
        patience=5,
        best_model_path="models/best_model_parte_03_01_SegNet.pth",
        dim=1
    )

    best_map = max([epoch_data[2].get('map', 0) for epoch_data in metrics])
    resultados_mAP[1].append(best_map)

print(f"\nResultados do Eixo 1:")
print(f"UNet: Media {np.mean(resultados_mAP[0])} Desvio Padrão {np.std(resultados_mAP[0])}")
print(f"SegNet: Media {np.mean(resultados_mAP[1])} Desvio Padrão {np.std(resultados_mAP[1])}")


resultados_mAP = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resultados_eixo2 = {
    'sem_peso': {0: [], 1: [], 2: [], 5: []},
    'com_peso': {0: [], 1: [], 2: [], 5: []}
}

# Auxiliares pra não ter que retreinar o modelo o tempo todo
global_best_map = 0.0
caminho_melhor_modelo = "models/best_model_parte_03_02.pth"

for balanceado in [False, True]:
    chave_peso = 'com_peso' if balanceado else 'sem_peso'

    # Calcula os pesos uma única vez por configuração
    if balanceado:
        weight = get_weights(loader=train_dataloader, classes=3, transform_mask=border_mask)
    else:
        weight = None

    for gamma in [0, 1, 2, 5]:

        for seed in [23, 24]:
            set_seed(seed)

            model = UNet(in_channels=3, out_channels=3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # gamma = 0 vira a Cross Entropy pura
            if gamma == 0:
                loss_fn = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fn = FocalLoss(weight=weight, gamma=gamma)

            metrics = train_model(
                model,
                loader_train=train_dataloader,
                loader_val=val_dataloader,
                optimizer=optimizer,
                epochs=50,
                loss=loss_fn,
                transform_mask=border_mask,
                decoder=multiclass_watershed_decoder,
                metric_function=compute_instance_metrics_wrapper,
                monitor_metric='map',
                stop_threshold=0.99,
                patience=10,
                best_model_path="models/best_model_parte_03_02.pth",
                dim=1
            )

            # Extrai o melhor mAP dessa rodada específica
            best_map_rodada = max([epoch_data[2].get('map', 0) for epoch_data in metrics])
            resultados_eixo2[chave_peso][gamma].append(best_map_rodada)

            if best_map_rodada > global_best_map:
                print(f"Modelo melhor encontrado: {best_map_rodada:.4f}")
                global_best_map = best_map_rodada
                torch.save(model.state_dict(), caminho_melhor_modelo)

print("\n==================================================================")

for balanceado in [False, True]:
    chave = 'com_peso' if balanceado else 'sem_peso'
    print(f"\n--- {chave.upper()} ---")

    # Gamma 0 (Cross Entropy)
    media_ce = np.mean(resultados_eixo2[chave][0])
    std_ce = np.std(resultados_eixo2[chave][0], ddof=1)
    print(f"CE: Média {media_ce:.4f} ± {std_ce:.4f}")

    # Focal Loss (Gammas 1, 2, 5)
    for gamma in [1, 2, 5]:
        media_fl = np.mean(resultados_eixo2[chave][gamma])
        std_fl = np.std(resultados_eixo2[chave][gamma], ddof=1)
        print(f"Focal (y={gamma}): Média {media_fl:.4f} ± {std_fl:.4f}")

print(f"\nO melhor modelo atingiu mAP de {global_best_map:.4f} e foi salvo em: {caminho_melhor_modelo}")


#########################################################################
#                           MODELO DA PARTE 5                           #
#########################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_final = ModUNet(in_channels=3, out_channels=3).to(device)
optimizer = torch.optim.Adam(model_final.parameters(), lr=1e-4)

loss_fn = FocalLoss(gamma=1)
# Treinamento
metrics = train_model(
    model_final,
    loader_train=train_dataloader,
    loader_val=val_dataloader,
    optimizer=optimizer,
    epochs=50,
    loss=loss_fn,
    transform_mask=border_mask,
    decoder=multiclass_watershed_decoder,
    metric_function=compute_instance_metrics_wrapper,
    monitor_metric='map',
    stop_threshold=0.99,
    patience=10,
    best_model_path="models/best_model_parte_05.pth",
    dim=1
)