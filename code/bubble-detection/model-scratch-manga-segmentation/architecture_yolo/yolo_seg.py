import torch.nn as nn
from .backbone import YSNBackbone, YSNNeck
from .blocks import ProtoNet
from .yolo_head import YSNHead

class YOLOSeg(nn.Module):
    """Model YOLO-Seg-Nano (YSN)"""
    def __init__(self, num_classes=2, num_prototypes=32):
        super().__init__()
        # Real number of class (no background)
        self.num_classes_obj = num_classes - 1 
        
        # --- HYPERPARAMETER TO FINE-TUNE MODEL ---
        base_channels = 16
        base_depth = 1
        
        # 1. Backbone
        self.backbone = YSNBackbone(base_channels=base_channels, base_depth=base_depth)
        
        # 2. Neck
        neck_channels = (base_channels * 8, base_channels * 8, base_channels * 16)
        self.neck = YSNNeck(*neck_channels)
        
        # 3. ProtoNet (take feature map shallowest from Neck)
        self.proto = ProtoNet(in_channels=neck_channels[0], num_prototypes=num_prototypes)
        
        # 4. Head
        self.head = YSNHead(num_classes=self.num_classes_obj, num_prototypes=num_prototypes, 
                            in_channels_list=neck_channels)

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        
        # Branch predict
        box_preds, cls_preds, mask_coef_preds = self.head(neck_features)
        
        # Branch create prototype mask
        proto_out = self.proto(neck_features[0])

        # Return all raw output
        # The processing logic will be outside the model (in the loss function or post-process)
        return box_preds, cls_preds, mask_coef_preds, proto_out