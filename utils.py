
import torch 
import torchvision
from dataset import SegementDataset
from torch.utils.data import DataLoader

"""
Functions for Unet segmentation 
"""

def save_checkpoint(state,filename="checkpoint.pth.tar"):
    print("Saving Chackpoint")
    torch.save(state,filename)
    
def load_checkpoint(checkpoint,model):
    print("Loading Chackpoint")
    model.load_state_dict(checkpoint["state_dict"])
    

def get_loader(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True 
    
): 
    
    """
    Getting data loader from data directories

    Returns:
        training and validation data loader
    """
    train_ds = SegementDataset(
        image_dir = train_dir,
        mask_dir=train_mask_dir,
        transforms=train_transform
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
         
    )
    

    val_ds = SegementDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transforms=val_transform
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
         
    )
    
    return train_loader , val_loader


def check_accuracy(loader,model,device="cuda"):
    """_summary_

    Args:
        loader (Data Loader): Load data
        model (UNET): UNET Model
        device (str, optional): cuda or cpu. Defaults to "cuda".

    Returns:
        accuracy
    """
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    
    acc = None
    
    for x,y in loader:
        x = x.to(device)
        y= y.to(device).unsqueeze(1)
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()
        num_correct += (preds == y).sum()
        num_pixels  += torch.numel(preds)
        dice_score += (2*(preds*y).sum())/((preds+y).sum()+1e-8)
        acc = num_correct/num_pixels*100
        
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f} ")
    print(f"Dice Score: {dice_score/len(loader)}")
 
    model.train()       
    return acc.to("cpu")    

def save_preds_as_imgs(
    loader,model,folder="saved_images/",device="cuda"   
):
    """_summary_

    Args:
        loader : (Data Loader)
        model : UNET model
        folder (str, optional): Images dierectory. Defaults to "saved_images/".
        device (str, optional): cuda or cpu. Defaults to "cuda".
    """
    model.eval()
    for idx,(x,y)   in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(y.unsqueeze(1),f"{folder}_{idx}.jpg")
            
    model.train()
            
            
            