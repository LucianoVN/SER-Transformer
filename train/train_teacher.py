import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.join(os.path.dirname(p=__file__), "../"))
from utils import CremaDataset, LogMelTransform, CustomViT, train_model_teacher

# define device (cuda if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

ANNOTATIONS_FILE = 'labels.csv'
AUDIO_DIR : str = os.path.join(os.path.dirname(p=__file__), f"../CREMA-D/")

traindataset = CremaDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            data_portion = 'train',
                            transform = LogMelTransform)

valdataset = CremaDataset(ANNOTATIONS_FILE,
                          AUDIO_DIR,
                          data_portion = 'validation',
                          transform = LogMelTransform)

testdataset = CremaDataset(ANNOTATIONS_FILE,
                           AUDIO_DIR,
                           data_portion = 'test',
                           transform = LogMelTransform)


# model
model = CustomViT().to(device)

# hyperparameters
lr = 1e-4
batch_size = 32
EPOCHS = 100
max_patience = 15
loss_fn = torch.nn.CrossEntropyLoss()

# filename for parameters
model_path = os.path.join(os.path.dirname(p=__file__), './train_results/teacher_params.pt')

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# dataloaders
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

# train model
results = train_model_teacher(model,
                              trainloader,
                              valloader,
                              optimizer,
                              loss_fn,
                              EPOCHS,
                              max_patience,
                              device,
                              model_path)