import os
import matplotlib.pyplot as plt
from torch_em.model import UNETR
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss  # MAE is L1Loss
import torch
from unetrdata import CentroidVectorData
import argparse

parser = argparse.ArgumentParser('UNETR tracking-training', add_help=False)


parser.add_argument('--use_loss_penalty', action='store_true', help='Apply a penalty map to the loss')
parser.set_defaults(norm_pix_loss=False)

args = parser.parse_args()

ds = CentroidVectorData("dfki/train")
testds = CentroidVectorData("dfki/train")

train_dataloader = DataLoader(ds, batch_size=8, shuffle=True)
test_dataloader = DataLoader(testds, batch_size=8, shuffle=False)

unetr_model = UNETR(img_size=448, backbone="mae", encoder="vit_l", encoder_checkpoint=None)
unetr_model.to("cuda")

optimizer = Adam(unetr_model.parameters(), lr=1e-5)
loss_fn = L1Loss()

num_epochs = 10000

for epoch in range(num_epochs):
    unetr_model.train()
    train_loss = 0.0
    #optimizer.zero_grad()
    for x, seg, vec, penalty_map in train_dataloader:
        x = x.to("cuda")
        seg = seg.to("cuda")
        vec = vec.to("cuda")
        penalty_map = penalty_map.to("cuda")
 
        #print("VECTOR SHAPE", vec.shape)
        y_pred = unetr_model(x)

        #y_pred = y_pred[:, :1, :, :]
        #print(y_pred.shape, seg.shape)
        if args.use_loss_penalty:
            loss = loss_fn(y_pred * penalty_map, seg)
        else:
            loss = loss_fn(y_pred, seg)

        #print(loss.item())
        train_loss += loss.item()
        
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss}")

    # Validation
    unetr_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, seg_val, vec_val, penalty_map_val in test_dataloader:
            x_val = x_val.to("cuda")
            seg_val = seg_val.to("cuda")
            vec_val = vec_val.to("cuda")
            penalty_map_val = penalty_map_val.to("cuda")
            
            y_pred_val = unetr_model(x_val)
            if args.use_loss_penalty:
                loss_val = loss_fn(y_pred_val * penalty_map_val, seg_val)
            else:
                loss_val = loss_fn(y_pred_val, seg_val)

            val_loss += loss_val.item()

    avg_val_loss = val_loss / len(test_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")

    # Save a validation sample with the current loss values as title
    val_sample, _m, _, _ = next(iter(test_dataloader))
    val_sample = val_sample.to("cuda")
    output = unetr_model(val_sample)
    output = output[0].detach().cpu()

    #print("plot shapes", val_sample[0].T.to("cpu").shape, output.squeeze().T.shape, _m[0].T.shape)
    fig, ax = plt.subplots(1,3)
    ax.flat[0].imshow(val_sample[0].T.to("cpu"))
    ax.flat[0].set_title("x")
    plt.suptitle(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    ax.flat[1].imshow(output.squeeze().T)
    ax.flat[1].set_title("y_pred")
    ax.flat[2].imshow(_m[0].T)
    ax.flat[2].set_title("y_target")
    plt.savefig(f"CENTROID_validation_sample_epoch_{epoch+1}.png")
    plt.clf()
    plt.close()
