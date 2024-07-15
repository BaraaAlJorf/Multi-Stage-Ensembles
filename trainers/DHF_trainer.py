from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.ehr_encoder import EHRTransformer
from models.cxr_encoder import CXRTransformer
from models.rr_encoder import RadiologyNotesEncoder
from models.dn_encoder import DischargeNotesEncoder
from models.classifier import MLPClassifier
from models.customtransformer import CustomTransformerLayer
from .trainer import Trainer
import pandas as pd


import numpy as np
from sklearn import metrics
import wandb

def relevancy_loss(y_fused_pred, y_true, r_scores, preds):
    # Calculate BCE for the fused prediction
    L_pred = F.binary_cross_entropy_with_logits(y_fused_pred, y_true)
    
    # Initialize total loss with L_pred
    total_loss = L_pred
    
    # Calculate and add L_{r_i} for each modality
    for modality in preds:
        y_pred = preds[modality]
        r_score = r_scores[modality]
        
        # BCE for the modality prediction
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        
        # L_{r_i} calculation
        L_r_i = ((bce_loss *r_score).abs() - 1) + bce_loss)
        total_loss += L_r_i
    
    return total_loss

class FusionTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl
        ):

        super(FusionTrainer, self).__init__(args)
        run = wandb.init(project=f'DHF_{self.args.fusion_type}', config=args)
        self.epoch = 0 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.token_dim = 384
        
        self.token_vector = torch.nn.Parameter(torch.randn(self.token_dim).to(self.device))
        token_vector_expanded = self.token_vector.unsqueeze(0).repeat(self.args.batch_size, 1)

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
        self.ehr_encoder = EHRTransformer(
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            dim_head=128
        )
        self.cxr_encoder = CXRTransformer(
            model_name='vit_small_patch16_384',
            image_size=384,
            patch_size=16,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            emb_dropout=0.0,
            dim_head=128
        )
        self.dn_encoder = DischargeNotesEncoder(
            pretrained_model_name='allenai/longformer-base-4096',
            output_dim=384
        )
        self.rn_encoder = RadiologyNotesEncoder(
            pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT',
            output_dim=384
        )
        
        self.ehr_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.cxr_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.rr_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.dn_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        
        self.ehr_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.cxr_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.rr_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.dn_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        
        self.final_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        
        # Initialize transformer layers
        self.transformer_layer1 = CustomTransformerLayer(input_dim=384*2, model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.transformer_layer2 = CustomTransformerLayer(input_dim=384*2, model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.transformer_layer3 = CustomTransformerLayer(input_dim=384*2, model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.transformer_layer4 = CustomTransformerLayer(input_dim=384*2, model_dim=384, nhead=4, num_layers=1).to(self.device)

        if self.args.mode == 'relevancy-based-hierarchical'::
            self.loss = relevancy_loss
            all_params = (
            list(self.ehr_encoder.parameters()) +
            list(self.cxr_encoder.parameters()) +
            list(self.dn_encoder.parameters()) +
            list(self.rn_encoder.parameters()) +
            list(self.ehr_r_classifier.parameters()) +
            list(self.cxr_r_classifier.parameters()) +
            list(self.dn_r_classifier.parameters()) +
            list(self.rr_r_classifier.parameters()) +
            list(self.ehr_classifier.parameters()) +
            list(self.cxr_classifier.parameters()) +
            list(self.dn_classifier.parameters()) +
            list(self.rr_classifier.parameters()) +
            list(self.transformer_layer1.parameters()) +
            list(self.transformer_layer2.parameters()) +
            list(self.transformer_layer3.parameters()) +
            list(self.transformer_layer4.parameters()) +
            [self.token_vector]
        )
        else:
            self.loss = nn.BCELoss()
            all_params = (
            list(self.ehr_encoder.parameters()) +
            list(self.cxr_encoder.parameters()) +
            list(self.dn_encoder.parameters()) +
            list(self.rn_encoder.parameters()) +
            list(self.transformer_layer1.parameters()) +
            list(self.transformer_layer2.parameters()) +
            list(self.transformer_layer3.parameters()) +
            list(self.transformer_layer4.parameters()) +
            [self.token_vector]
        )

        
        self.optimizer = optim.Adam(all_params, lr=args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
    
    def set_train_mode(self):
        """Set all neural network components to training mode."""
        self.ehr_encoder.train()
        self.cxr_encoder.train()
        self.dn_encoder.train()
        self.rr_encoder.train()

        self.ehr_r_classifier.train()
        self.cxr_r_classifier.train()
        self.dn_r_classifier.train()
        self.rr_r_classifier.train()

        self.ehr_classifier.train()
        self.cxr_classifier.train()
        self.dn_classifier.train()
        self.rr_classifier.train()

        self.transformer_layer1.train()
        self.transformer_layer2.train()
        self.transformer_layer3.train()
        self.transformer_layer4.train()

    def set_eval_mode(self):
        """Set all neural network components to evaluation mode."""
        self.ehr_encoder.eval()
        self.cxr_encoder.eval()
        self.dn_encoder.eval()
        self.rr_encoder.eval()

        self.ehr_r_classifier.eval()
        self.cxr_r_classifier.eval()
        self.dn_r_classifier.eval()
        self.rr_r_classifier.eval()

        self.ehr_classifier.eval()
        self.cxr_classifier.eval()
        self.dn_classifier.eval()
        self.rr_classifier.eval()

        self.transformer_layer1.eval()
        self.transformer_layer2.eval()
        self.transformer_layer3.eval()
        self.transformer_layer4.eval()
    
    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, dn, rn, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)
            
            vectors = {}
            r_scores = {}

            if 'ehr' in self.args.modalities:
                v_ehr = self.ehr_encoder(x)
                vectors['ehr'] = v_ehr
                r_ehr = self.ehr_r_classifier(v_ehr)
                r_scores['ehr'] = r_ehr
                y_ehr_pred = self.ehr_classifier(v_ehr)
                preds['ehr'] = y_ehr_pred
            if 'cxr' in self.args.modalities:
                v_cxr = self.cxr_encoder(img)
                vectors['cxr'] = v_cxr
                r_cxr = self.cxr_r_classifier(v_cxr)
                r_scores['cxr'] = r_cxr
                y_cxr_pred = self.cxr_classifier(v_cxr)
                preds['cxr'] = y_cxr_pred
            if 'dn' in self.args.modalities:
                v_dn = self.dn_encoder(dn)
                vectors['dn'] = v_dn
                r_dn = self.dn_r_classifier(v_dn)
                r_scores['dn'] = r_dn
                y_dn_pred = self.dn_classifier(v_dn)
                preds['dn'] = y_dn_pred
            if 'rr' in self.args.modalities:
                v_rr = self.rr_encoder(rr)
                vectors['rr'] = v_rr
                r_rr = self.rr_r_classifier(v_rr)
                r_scores['rr'] = r_rr
                y_rr_pred = self.rr_classifier(v_rr)
                preds['rr'] = y_rr_pred

            if self.args.mode == 'relevancy-based-hierarchical':
                modalities_list = self.args.modalities.split('-')
                scores_tensor = torch.stack([torch.tensor(r_scores[mod]) for mod in modalities_list], dim=1)
                sorted_scores, sorted_indices = torch.sort(scores_tensor, dim=1, descending=True)
                vectors_tensor = torch.stack([v_ehr, v_cxr, v_dn, v_rr], dim=1)
                first_priority = vectors_tensor[torch.arange(self.args.batch_size), sorted_indices[:, 0]]
                #sorted_modalities = sorted(r_scores, key=r_scores.get)
                fused_vector = torch.cat((first_priority, self.token_vector_expanded), dim=1)
                
            elif self.args.mode == 'predefined-hierarchical' and self.args.order is not None:
                # Use predefined order
                order_list = self.args.order.split('-')
                sorted_modalities = [mod for mod in order_list if mod in r_scores]
                fused_vector = torch.cat((vectors[sorted_modalities[0]], self.token_vector_expanded), dim=1)


            # Dynamically use different transformer layers for each modality combination
            for idx, modality in enumerate(sorted_modalities[1:], 1):
                fused_vector = torch.cat((fused_vector, vectors[modality]), dim=1)
                transformer_layer = getattr(self, f'transformer_layer{idx}')
                fused_vector = transformer_layer(fused_vector)

            # Final classifier
            y_fused_pred = self.final_classifier(fused_vector)
            
            if self.args.mode == 'relevancy-based-hierarchical':
                loss = self.loss(y_fused_pred, y, r_scores, preds)
            else:
                loss = self.loss(y_fused_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            outPRED = torch.cat((outPRED, y_fused_pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
        

        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        wandb.log({
                'train_Loss': epoch_loss/i, 
                'train_AUC': ret['auroc_mean']
            })
        return ret
    
    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
    
        with torch.no_grad():
            for i, (x, img, dn, rn, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = x.to(self.device)
                y = y.to(self.device)
                img = img.to(self.device)
                
                vectors = {}
                r_scores = {}
    
                if 'ehr' in self.args.modalities:
                    v_ehr = self.ehr_encoder(x)
                    vectors['ehr'] = v_ehr
                    r_ehr = self.ehr_r_classifier(v_ehr)
                    r_scores['ehr'] = r_ehr
                    y_ehr_pred = self.ehr_classifier(v_ehr)
                    preds['ehr'] = y_ehr_pred
                if 'cxr' in self.args.modalities:
                    v_cxr = self.cxr_encoder(img)
                    vectors['cxr'] = v_cxr
                    r_cxr = self.cxr_r_classifier(v_cxr)
                    r_scores['cxr'] = r_cxr
                    y_cxr_pred = self.cxr_classifier(v_cxr)
                    preds['cxr'] = y_cxr_pred
                if 'dn' in self.args.modalities:
                    v_dn = self.dn_encoder(dn)
                    vectors['dn'] = v_dn
                    r_dn = self.dn_r_classifier(v_dn)
                    r_scores['dn'] = r_dn
                    y_dn_pred = self.dn_classifier(v_dn)
                    preds['dn'] = y_dn_pred
                if 'rr' in self.args.modalities:
                    v_rr = self.rr_encoder(rr)
                    vectors['rr'] = v_rr
                    r_rr = self.rr_r_classifier(v_rr)
                    r_scores['rr'] = r_rr
                    y_rr_pred = self.rr_classifier(v_rr)
                    preds['rr'] = y_rr_pred
    
                if self.args.mode == 'relevancy-based-hierarchical':
                    modalities_list = self.args.modalities.split('-')
                    scores_tensor = torch.stack([torch.tensor(r_scores[mod]) for mod in modalities_list], dim=1)
                    sorted_scores, sorted_indices = torch.sort(scores_tensor, dim=1, descending=True)
                    vectors_tensor = torch.stack([v_ehr, v_cxr, v_dn, v_rr], dim=1)
                    first_priority = vectors_tensor[torch.arange(self.args.batch_size), sorted_indices[:, 0]]
                    #sorted_modalities = sorted(r_scores, key=r_scores.get)
                    fused_vector = torch.cat((first_priority, self.token_vector_expanded), dim=1)
                    
                elif self.args.mode == 'predefined-hierarchical' and self.args.order is not None:
                    # Use predefined order
                    order_list = self.args.order.split('-')
                    sorted_modalities = [mod for mod in order_list if mod in r_scores]
                    fused_vector = torch.cat((vectors[sorted_modalities[0]], self.token_vector_expanded), dim=1)
    
    
                # Dynamically use different transformer layers for each modality combination
                for idx, modality in enumerate(sorted_modalities[1:], 1):
                    fused_vector = torch.cat((fused_vector, vectors[modality]), dim=1)
                    transformer_layer = getattr(self, f'transformer_layer{idx}')
                    fused_vector = transformer_layer(fused_vector)
    
                # Final classifier
                y_fused_pred = self.final_classifier(fused_vector)
                
                if self.args.mode == 'relevancy-based-hierarchical':
                    loss = self.loss(y_fused_pred, y, r_scores, preds)
                else:
                    loss = self.loss(y_fused_pred, y)
                epoch_loss += loss.item()
                outPRED = torch.cat((outPRED, y_fused_pred), 0)
                outGT = torch.cat((outGT, y), 0)
    
            print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
    
            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy())
            np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy())
            self.epochs_stats['auroc val'].append(ret['auroc_mean'])
            self.epochs_stats['loss val'].append(epoch_loss / i)
            wandb.log({
                'val_Loss': epoch_loss / i,
                'val_AUC': ret['auroc_mean']
            })
    
        return ret

    def eval(self):

        if self.args.fusion_type != 'late_avg':
            self.load_ehr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
            self.load_cxr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
            self.load_state(state_path=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
        
        self.epoch = 0
        self.model.eval()

        ret = self.validate(self.test_dl)
        wandb.log({
                'test_auprc': ret['auprc_mean'], 
                'test_AUC': ret['auroc_mean']
            })
            if self.args.task!="length-of-stay":
                self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
        return
    
    def train(self):
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.set_eval_mode() 
            ret = self.validate(self.val_dl)
            #self.save_checkpoint(prefix='last')
    
            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_checkpoint()
                self.patience = 0
            else:
                self.patience += 1
    
            self.set_train_mode() 
            self.train_epoch()
            
            if self.patience >= self.args.patience:
                break

        
    

