import torch
from pathlib import Path

from psm_final.analysis.model import ModelAnalysisBase
from psm_final.models.beta_vae import _image_transform
from psm_final.models.cnn_basemodel import SimpleCNN, IMAGE_SIZE

from torchvision import transforms

from psm_final.models.cnn_resnet50 import build_resnet_model


class CNNAnalysis(ModelAnalysisBase):
    CHECKPOINT_GLOB = "results/cnn_basemodel/*.pth"

    @classmethod
    def discover(cls, *, triple_n_path, checkpoints_root, device=None):
        specs = []
        for ckpt in Path(checkpoints_root).glob(cls.CHECKPOINT_GLOB):
            specs.append((f"CNN ({ckpt.stem})", str(ckpt)))
        specs.sort(key=lambda spec: spec[0])
        return [
            (label, lambda path=path: cls(triple_n_path=triple_n_path,
                                          model_path=path, device=device))
            for label, path in specs
        ]

    def __init__(self, triple_n_path, model_path, device=None):
        super().__init__(triple_n_path)

        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_classes = checkpoint.get('num_classes', state_dict['fc2.weight'].shape[0])
            # Embed at the resolution the model was trained on (stored in the
            # checkpoint). Falls back to IMAGE_SIZE for older checkpoints that
            # predate this field, preserving their original 64px behaviour.
            image_size = checkpoint.get('image_size', IMAGE_SIZE)
        else:
            state_dict = checkpoint
            num_classes = state_dict['fc2.weight'].shape[0]
            image_size = IMAGE_SIZE

        self.model = SimpleCNN(num_classes=num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = _image_transform(image_size)

    def embedding(self, image):
        image_tensor = self.transform(image.convert('RGB')).float().div(255.0)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, features = self.model(image_tensor, return_features=True)

        return features.squeeze(0).cpu().numpy()

class ResNet50Analysis(ModelAnalysisBase):
    CHECKPOINT_GLOB = "results/cnn_resnet50/*.pth"

    @classmethod
    def discover(cls, *, triple_n_path, checkpoints_root, device=None):
        specs = []
        for ckpt in Path(checkpoints_root).glob(cls.CHECKPOINT_GLOB):
            specs.append((f"ResNet50 ({ckpt.stem})", str(ckpt)))
        specs.sort(key=lambda spec: spec[0])
        return [
            (label, lambda path=path: cls(triple_n_path=triple_n_path,
                                          model_path=path, device=device))
            for label, path in specs
        ]

    def __init__(self, triple_n_path, model_path, device=None):
        super().__init__(triple_n_path)

        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # cnn_resnet50.py always saves the metadata-wrapped format, so no
        # bare-state-dict fallback is needed here (unlike CNNAnalysis, which
        # supports older checkpoints saved before that format existed).
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = build_resnet_model(num_classes=checkpoint['num_classes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # ResNet50 has no return_features flag like SimpleCNN -- it's fixed,
        # third-party code. A forward hook captures avgpool's output (the
        # 2048-dim pooled features right before the final classifier) during
        # a normal forward pass, without modifying the model at all.
        self._features = None
        self.model.avgpool.register_forward_hook(self._capture_features)

        # ImageNet-standard preprocessing -- resnet50 was pretrained at 224x224
        # with these exact normalization stats, unlike SimpleCNN's own 32px
        # from-scratch pipeline (_image_transform).
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _capture_features(self, module, input, output):
        self._features = output

    def embedding(self, image):
        image_tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model(image_tensor)   # triggers the hook; classifier output itself unused

        return self._features.flatten().cpu().numpy()   # (2048,) embedding for this image