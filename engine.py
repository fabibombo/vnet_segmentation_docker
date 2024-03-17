import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from typing import Dict, List, Tuple
import gc

import pandas as pd
from tqdm.auto import tqdm

import utils
import metrics

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    criterion: torch.nn.Module, 
    device: torch.device,
    nll: bool = False,
) -> Dict[str, float]:
    
    model.train()

    train_loss = 0
    train_metrics = {"dice": 0, "jaccard": 0, "bcloss": 0, "recall": 0}
    
    # torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

        y_probs = model(X.to(device)).to("cpu")

        # if torch.any(torch.isnan(y_probs)):
        #     print("Batch:", batch)
        #     print("NaN alert!")
        #     print(torch.isnan(X).any(), torch.isnan(y).any(), torch.isnan(y_probs).any())
        #     torch.save({"X": X, "y": y, "preds": y_probs}, "troubleshoot_nans.pt")
        
        if nll:
            loss = criterion(y_probs, y.view(-1).long())
        else:
            y_probs = y_probs[:,1].clamp(0,1)
            loss = criterion(y_probs, y.view(-1)) #clamp bc we have some errors, let's hope they are not nans

        train_loss += loss.detach().item()

        optimizer.zero_grad()
        
        loss.backward(retain_graph=None)

        # add gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
        
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        if nll:
            metric_bcloss = F.nll_loss(y_probs, y.view(-1).long()).detach().item() # flatten target for loss
            y_probs = F.softmax(y_probs, dim=1)
            y_probs = y_probs[:,1].clamp(0,1)
        else:
            #metric_bcloss = F.binary_cross_entropy(y_probs[:,1], y.view(-1)).detach().item()
            metric_bcloss = F.binary_cross_entropy(y_probs, y.view(-1)).detach().item()
        
        y_probs = y_probs.view(y.shape)
    
        metric_dice = metrics.dice(y_probs, y).detach().item() 
        metric_jaccard = metrics.jaccard_index(y_probs, y).detach().item() 
        metric_recall = metrics.recall(y_probs, y, device).detach().item() 
        
        train_metrics["bcloss"] += metric_bcloss
        train_metrics["dice"] += metric_dice
        train_metrics["jaccard"] += metric_jaccard
        train_metrics["recall"] += metric_recall

        #cleaning memory
        gc.collect()
        torch.cuda.empty_cache()
            
        ## print every 20%
        if batch != 0 and batch % (len(dataloader) // 5) == 0 and batch < (9 * (len(dataloader) // 10)):
            print(
                "\t",
                batch,
                "\t",
                "Loss:",
                round(train_loss / (batch + 1), 4),
                "BCLoss:",
                round(train_metrics["bcloss"] / (batch + 1), 4),
                "Dice:",
                round(train_metrics["dice"] / (batch + 1), 4),
                "Jaccard:",
                round(train_metrics["jaccard"] / (batch + 1), 4),
                "Recall:",
                round(train_metrics["recall"] / (batch + 1), 4),
            )

    for key in train_metrics:
        train_metrics[key] /= len(dataloader)

    return {"loss": train_loss / len(dataloader), **train_metrics}

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module, 
    device: torch.device,
    nll: bool = False,
) -> Dict[str, float]:
    model.eval()

    test_loss = 0
    test_metrics = {"dice": 0, "jaccard": 0, "bcloss": 0, "recall": 0}

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            #X, y = X.to(device), y.to(device)
    
            # flatten y for nll
            # y = y.view(-1)
            y_probs = model(X.to(device)).to("cpu")

            if nll:
                loss = criterion(y_probs, y.view(-1).long()).detach().item()
            else:
                y_probs = y_probs[:,1].clamp(0,1)
                #loss = criterion(y_probs[:,1], y.view(-1)).detach().item()
                loss = criterion(y_probs, y.view(-1)).detach().item()
                
            test_loss += loss

            if nll:
                metric_bcloss = F.nll_loss(y_probs, y.view(-1).long()).detach().item() # flatten target for loss
                y_probs = F.softmax(y_probs, dim=1)
                y_probs = y_probs[:,1]
            else:
                metric_bcloss = F.binary_cross_entropy(y_probs, y.view(-1)).detach().item()
            
            y_probs = y_probs.view(y.shape)
            
            metric_dice = metrics.dice(y_probs, y).detach().item() 
            metric_jaccard = metrics.jaccard_index(y_probs, y).detach().item() 
            metric_recall = metrics.recall(y_probs, y, device).detach().item() 
            
            test_metrics["bcloss"] += metric_bcloss
            test_metrics["dice"] += metric_dice
            test_metrics["jaccard"] += metric_jaccard
            test_metrics["recall"] += metric_recall

            gc.collect()
            torch.cuda.empty_cache()
        
        test_loss /= len(dataloader)
        for key in test_metrics:
            test_metrics[key] /= len(dataloader)

    return {"loss": test_loss, **test_metrics}

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    criterion: torch.nn.Module, 
    epochs: int,
    results_path: str,
    models_path: str,
    device: torch.device,
    nll: bool = False,
) -> Dict[str, List]:
    
    results = {
        "epoch": [],
        "train_loss": [],
        "train_bcloss": [],
        "train_dice": [],
        "train_jaccard": [],
        "train_recall": [],
        "test_loss": [],
        "test_bcloss": [],
        "test_dice": [],
        "test_jaccard": [],
        "test_recall": [],
    }

    min_test_metric = None
    
    print(" ## BEGIN TRAINING ## ")
    print("    Model:                 \t", model.name)
    print("    Number of train batches:\t", len(train_dataloader))
    print("    Number of test batches:\t", len(test_dataloader))

    for epoch in tqdm(range(epochs)):
        train_results = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            device=device,
            nll=nll
        )
        test_results = test_step(
            model=model, 
            criterion=criterion,
            dataloader=test_dataloader, 
            device=device,
            nll=nll
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_results['loss']:.4f} | "
            f"train_bcloss: {train_results['bcloss']:.4f} | "
            f"train_dice: {train_results['dice']:.4f} | "
            f"train_jaccard: {train_results['jaccard']:.4f} | "
            f"train_recall: {train_results['recall']:.4f}\n\t |  "
            f"test_loss: {test_results['loss']:.4f} |  "
            f"test_bcloss: {test_results['bcloss']:.4f} |  "
            f"test_dice: {test_results['dice']:.4f} |  "
            f"test_jaccard: {test_results['jaccard']:.4f}  | "
            f"test_recall: {test_results['recall']:.4f}"
        )

        results["epoch"].append(epoch)
        results["train_loss"].append(train_results["loss"])
        results["train_bcloss"].append(train_results["bcloss"])
        results["train_dice"].append(train_results["dice"])
        results["train_jaccard"].append(train_results["jaccard"])
        results["train_recall"].append(train_results["recall"])
        results["test_loss"].append(test_results["loss"])
        results["test_bcloss"].append(test_results["bcloss"])
        results["test_dice"].append(test_results["dice"])
        results["test_jaccard"].append(test_results["jaccard"])
        results["test_recall"].append(test_results["recall"])

        write_results(results, results_path +"results_" + model.name + ".csv")

        if min_test_metric is None:
            min_test_metric = test_results["dice"]
        elif epoch > 20 and test_results["dice"] > min_test_metric:
            min_test_metric = test_results["dice"]
            utils.save_model(model=model,
                           target_dir=models_path,
                           model_name=model.name + "_weights_min_test_metric.pth")

        #save the last trained epoch with overwritting in case the training stops and we need to resume it
        utils.save_model(model=model,
                           target_dir=models_path,
                           model_name=model.name + "_last_trained_epoch.pth")
    return results

def write_results(results, csv_path):
    # we send everything to cpu just in case
    results_cpu = {}
    for key, value in results.items():
        tmp_list = []
        for item in results[key]:
            if torch.is_tensor(item):
                tmp_list.append(item.cpu().detach().numpy())
            else:
                tmp_list.append(item)
            results_cpu[key] = tmp_list

    df = pd.DataFrame(results_cpu)
    df.to_csv(csv_path, index=False)
