import jittor as jt
from jittor.models import resnet50, Resnext50_32x4d, resnet18, resnet101, vgg16_bn
# from resnet import resnet50, Resnext50_32x4d, resnet18, resnet101

def get_model(num_classes=6):
    model = resnet50(pretrained=True)
    # model = resnet18(pretrained=True)
    # model = Resnext50_32x4d(pretrained=True)
    # model = vgg16_bn(pretrained=True)
    model.fc = jt.nn.Linear(model.fc.in_features, num_classes)
    return model
