import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, conf, num_scales=4):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            scale_conf = self._adjust_conf_for_scale(conf, i)
            discriminator = SingleScaleDiscriminator(scale_conf, scale_id=i)
            self.discriminators.append(discriminator)
        
        self.beta = nn.Parameter(torch.ones(num_scales)) 
        
        self.register_buffer('acc_history', torch.ones(num_scales) * 0.5)
        self.momentum = 0.9 
        
    def _adjust_conf_for_scale(self, conf, scale_id):

        scale_conf = type('', (), {})()
        for attr in dir(conf):
            if not attr.startswith('_'):
                setattr(scale_conf, attr, getattr(conf, attr))
        
        if scale_id == 0:
            scale_conf.input_size = getattr(conf, 'input_size', 128)
            scale_conf.ndf = getattr(conf, 'ndf', 64)
        elif scale_id == 1: 
            scale_conf.input_size = getattr(conf, 'input_size', 128) // 2
            scale_conf.ndf = getattr(conf, 'ndf', 64) * 2
        elif scale_id == 2:
            scale_conf.input_size = getattr(conf, 'input_size', 128) // 4
            scale_conf.ndf = getattr(conf, 'ndf', 64) * 4
        else:  
            scale_conf.input_size = getattr(conf, 'input_size', 128) // 8
            scale_conf.ndf = getattr(conf, 'ndf', 64) * 8
            
        return scale_conf
    
    def forward(self, x):

        predictions = []
        features = []
        

        for i, discriminator in enumerate(self.discriminators):
            if i == 0: 
                scale_input = x
            elif i == 1: 
                scale_input = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            elif i == 2:  
                scale_input = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
            else:  
                scale_input = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
            
            pred, feat = discriminator(scale_input)
            predictions.append(pred)
            features.append(feat)
        
        return predictions, features
    
    def compute_adaptive_weights(self, accuracies):


        self.acc_history = self.momentum * self.acc_history + (1 - self.momentum) * accuracies
        

        weighted_acc = self.beta * self.acc_history
        weights = F.softmax(weighted_acc, dim=0)
        
        return weights
    
    def compute_hdm_loss(self, predictions_real, features_real, 
                        predictions_fake, features_fake, 
                        real_labels, fake_labels, lambda_feature=1.0):

        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        
        accuracies = []
        scale_losses = []
        
        for i in range(self.num_scales):
            loss_real = bce_loss(predictions_real[i], real_labels)
            loss_fake = bce_loss(predictions_fake[i], fake_labels)
            adv_loss = 0.5 * (loss_real + loss_fake)
            
            feature_loss = mse_loss(features_real[i], features_fake[i])
            
            scale_loss = adv_loss + lambda_feature * feature_loss
            scale_losses.append(scale_loss)
            
            with torch.no_grad():
                pred_real_binary = (predictions_real[i] > 0.5).float()
                pred_fake_binary = (predictions_fake[i] < 0.5).float()
                acc_real = pred_real_binary.mean()
                acc_fake = pred_fake_binary.mean()
                acc = 0.5 * (acc_real + acc_fake)
                accuracies.append(acc)
        
        accuracies = torch.stack(accuracies)
        scale_losses = torch.stack(scale_losses)

        adaptive_weights = self.compute_adaptive_weights(accuracies)
        
        total_loss = torch.sum(adaptive_weights * scale_losses)
        
        return total_loss, adaptive_weights, accuracies


class SingleScaleDiscriminator(nn.Module):
    def __init__(self, conf, scale_id=0):
        super(SingleScaleDiscriminator, self).__init__()
        self.scale_id = scale_id
        
        model = self._build_encoder(conf)
        layers = list(model.children())
        
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(
            layers[-1],
            nn.Sigmoid()
        )
        
        self._replace_relu_with_leaky_relu()
    
    def _build_encoder(self, conf):
        ndf = getattr(conf, 'ndf', 64)
        nc = getattr(conf, 'nc', 4) 
        
        if self.scale_id == 0: 
            layers = [
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            ]
        elif self.scale_id == 3: 
            layers = [
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
            ]
        else:  
            layers = [
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ndf * 4, 1, 1, 1, 0, bias=False),
            ]
        
        return nn.Sequential(*layers)
    
    def _replace_relu_with_leaky_relu(self):
        def replace_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.LeakyReLU(0.2, inplace=True))
                else:
                    replace_relu(child)
        
        replace_relu(self.features)
    
    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(x.size(0), -1).mean(dim=1)
        return classifier, features


