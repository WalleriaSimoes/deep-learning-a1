# Programming Assignment 1

## Ambiente e Dependências
Para instalar todas as dependências necessárias para o projeto, execute o seguinte comando na raiz do repositório:

```
pip install -r requirements.txt
```

## Download dos Dados
Boa parte do projeto foi desenvolvida utilizando o Google Colab. O download dos dados da competição (BBBC038) para o ambiente de execução pode ser feito executando o script abaixo:

```
import os
os.environ['KAGGLE_API_TOKEN'] = "SUA_CHAVE_AQUI"

# 1. Baixa o arquivo principal da competição
!kaggle competitions download -c data-science-bowl-2018

# 2. Descompacta o arquivo principal em uma pasta temporária
!unzip -q data-science-bowl-2018.zip -d dataset_temp/

# 3. Descompacta APENAS o stage1_train.zip na pasta final
!unzip -q dataset_temp/stage1_train.zip -d stage1_train/

# 4. Limpa os arquivos pesados para liberar espaço
!rm data-science-bowl-2018.zip
!rm -rf dataset_temp/
```

De toda forma, eles também estão disponíveis na pasta ``data/`` do repositório para uso local

## Treinamento e Avaliação

Para treinar o modelo com as melhores configurações encontradas no script.ipynb, execute o seguinte comando na raiz do repositório:

```
python train.py
```

Para avaliar o modelo treinado (calculando o mAP no conjunto de validação), utilize:

```
python evaluate.py
```

## Inferência
Caso queira utilizar o modelo já treinado (.pth) para obter a máscara de instâncias de uma imagem específica, abra o arquivo ´inferencia.ipynb´. Ele carrega os pesos finais e realiza a predição e a contagem de células de forma isolada, sem necessidade de retreino.