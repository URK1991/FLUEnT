import torch
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy
import random

#Data augmentation and transformation
def get_transforms(split):
    print('getting data transforms')
    data_transforms = {
    'Train': [
        transforms.RandomResizedCrop(size=(256, 256),scale=(0.9, 1.0),ratio=(9 / 10, 10 / 9)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(23)], p=0.8),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.125)], p=0.8),
        transforms.ToTensor()
        ],
    'Val': [
        transforms.Resize((256,256)),
        transforms.ToTensor()
        ]
    }
    return transforms.Compose(data_transforms[split])

#Checking the availability of the computational resource
def device_avail():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    return device

#CNN VAE encoding the input image to a matrix of size 26x26
class CNNVAE(nn.Module):
    def __init__(self, input_dim, dim1,dim2,dim3,dim4, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim1, 4, 2, 1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(True),
            nn.Conv2d(dim1, dim2, 4, 2, 1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(True),
            nn.Conv2d(dim2, dim3, 4, 2, 1),
            nn.BatchNorm2d(dim3),
            nn.ReLU(True),
            nn.Conv2d(dim3, dim4, 5, 1, 0),
            nn.BatchNorm2d(dim4),
            nn.ReLU(True),
            nn.Conv2d(dim4, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )
        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim4, 3, 1, 0),
            nn.BatchNorm2d(dim4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim4, dim3, 5, 1, 0),
            nn.BatchNorm2d(dim3),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim3, dim2, 4, 2, 1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim2, dim1, 4, 2, 1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim1, input_dim, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1) #Extracting the mean and variance from the z_dim*2

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean() #calculating KLD for the given mu and logvar
        lat_vec = q_z_x.rsample() #Sampling latent vector from the latent space of mean (mu) and variance (logvar)
        x_tilde = self.decoder(lat_vec) #Image reconstruction 
        return x_tilde,kl_div
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32   #Seed for reproducibility of experiments
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    
def train_model():
    g = torch.Generator()
    g.manual_seed(0)
    data_dir = '/projectnb/ivcgroup/ukhan3/PData' #Data/Source Directory that can be taken as an argument from main
    device = device_avail()
    model = CNNVAE(3,64,128,256,512,1).to(device)  
    
    dataloaders = {};
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),get_transforms(x)) for x in ['Train', 'Val']}
    dataloaders['Train'] = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=16,shuffle=True, num_workers=3,worker_init_fn=seed_worker, generator=g)

    dataloaders['Val'] = torch.utils.data.DataLoader(image_datasets['Val'], batch_size=16,
                                             shuffle=False, num_workers=3)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Val']}
    

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3, weight_decay=1e-4)
    
    val_losses = []
    train_losses = []
    
    the_last_loss = 100000
 
    low_loss = the_last_loss

    num_epochs = 150 #Set to 150 for the experiment but can be set an argument for the user to take as input
    kld_weight = 1 #Set to one for the experiment but can be set an argument for the user to take as input
    recon_weight = 1 #Set to one for the experiment but can be set an argument for the user to take as input
   
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['Train', 'Val']:
                if phase =='Train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                
                for img, target in dataloaders[phase]:
                    img = img.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()
 
                    with torch.set_grad_enabled(phase == 'Train'): 
                        x_tilde, kl_d = model(img)
                        
                        loss_recons = F.mse_loss(x_tilde, img, size_average=False) / img.size(0)

                        loss = loss_recons*recon_weight + kl_d*kld_weight
                    
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()

                        
                            
                    running_loss += loss.item() * img.size(0)

                epoch_loss = running_loss / dataset_sizes[phase]
                
                if phase == 'Train':
                    train_losses.append(epoch_loss)
                   
                else:
                    val_losses.append(epoch_loss) 
                
                
                #Saving the best model in case of the lowest loss in the training epochs (No early stopping was used)
                if phase == 'Val' and epoch_loss < low_loss:
                    print('The best loss of VAE is', epoch_loss)
                    low_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                                        
                if phase == 'Val':
                    the_last_loss = epoch_loss
                    print('The epoch loss of VAE is', epoch_loss)
                
                #Saving the models after every 30 epochs
                if (epoch+1)%30 == 0:
                    v = epoch+1
                    torch.save(best_model_wts, '/projectnb/ivcgroup/ukhan3/NIPS_OUT/' + '26x26_MSEKLD' + str(v) + '.pth') #saving directory that can be taken as argument
                
   
   
    model.load_state_dict(best_model_wts) 
    return model

if __name__ == "__main__":
    print('26x26 VAE')
    tr_model = train_model()
    torch.save(tr_model, '/projectnb/ivcgroup/ukhan3/output/' + '26x26_MSEKLD.pth') #saving directory that can be taken as argument
