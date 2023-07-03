from models import *


def build_img_model(args):
    """
    This function is used for building image model, then return the model
    """
    net, netT = None, None

    if args.model == 'resnet18':
        net = ResNet18()
    elif args.model == 'resnet34':
        net = ResNet34()
    elif args.model == 'resnet50':
        net = ResNet50()
    elif args.model == 'resnet101':
        net = ResNet101()
    elif args.model == 'resnet152':
        net = ResNet152()
    elif args.model == 'densenet121':
        net = DenseNet121()
    elif args.model == 'densenet169':
        net = DenseNet169()
    elif args.model == 'densenet201':
        net = DenseNet201()
    elif args.model == 'densenet161':
        net = DenseNet161()
    elif args.model == 'deit_tiny_patch16_224':
        net = deit_tiny_patch16_224()
    elif args.model == 'deit_small_patch16_224':
        net = deit_small_patch16_224()
    elif args.model == 'deit_base_patch16_224':
        net = deit_base_patch16_224()
    elif args.model == 'deit_tiny_distilled_patch16_224':
        net = deit_tiny_distilled_patch16_224()
    elif args.model == 'deit_small_distilled_patch16_224':
        net = deit_small_distilled_patch16_224()
    elif args.model == 'deit_base_distilled_patch16_224':
        net = deit_base_distilled_patch16_224()
    elif args.model == 'deit_tiny_patch16_224_linear':
        net = deit_tiny_patch16_224_linear()
    elif args.model == 'deit_small_patch16_224_linear':
        net = deit_small_patch16_224_linear()
    elif args.model == 'deit_base_patch16_224_linear':
        net = deit_base_patch16_224_linear()
    elif args.model == 'deit_tiny_distilled_patch16_224_linear':
        net = deit_tiny_distilled_patch16_224_linear()
    elif args.model == 'deit_small_distilled_patch16_224_linear':
        net = deit_small_distilled_patch16_224_linear()
    elif args.model == 'deit_base_distilled_patch16_224_linear':
        net = deit_base_distilled_patch16_224_linear()
    elif args.model == 'swin_tiny_patch4_window7_224':
        net = swin_tiny_patch4_window7_224()
    elif args.model == 'swin_small_patch4_window7_224':
        net = swin_small_patch4_window7_224()
    elif args.model == 'swin_base_patch4_window7_224':
        net = swin_base_patch4_window7_224()
    elif args.model == 'swin_large_patch4_window7_224_linear':
        netT = swin_large_patch4_window7_224_linear()
        net = swin_large_patch4_window7_224_linear()
    elif args.model == 'resnet18_linear':
        net = ResNet18_Linear()
    elif args.model == 'resnet34_linear':
        net = ResNet34_Linear()
    elif args.model == 'resnet50_linear':
        net = ResNet50_Linear()
    elif args.model == 'resnet101_linear':
        net = ResNet101_Linear()
    elif args.model == 'resnet152_linear':
        net = ResNet152_Linear()
    elif args.model == 'efficientnet_b0_linear':
        net = EfficientNet_B0_Linear()
    elif args.model == 'efficientnet_b1_linear':
        net = EfficientNet_B1_Linear()
    elif args.model == 'efficientnet_b2_linear':
        net = EfficientNet_B2_Linear()
    elif args.model == 'efficientnet_b3_linear':
        net = EfficientNet_B3_Linear()
    elif args.model == 'efficientnet_b4_linear':
        net = EfficientNet_B4_Linear()
    elif args.model == 'efficientnet_b5_linear':
        net = EfficientNet_B5_Linear()
    elif args.model == 'efficientnet_b6_linear':
        net = EfficientNet_B6_Linear()
    elif args.model == 'efficientnet_b7_linear':
        net = EfficientNet_B7_Linear()
    elif args.model == 'efficientnetv2_s_linear':
        net = EfficientNetV2_S_Linear()
    elif args.model == 'efficientnetv2_m_linear':
        net = EfficientNetV2_M_Linear()
    elif args.model == 'efficientnetv2_l_linear':
        net = EfficientNetV2_L_Linear()
    elif args.model == 'efficientnetv2_xl_linear':
        net = EfficientNetV2_XL_Linear()
    elif args.model == 'seresnext26d_32x4d':
        net = seresnext26d_32x4d()
    elif args.model == 'seresnext26t_32x4d':
        net = seresnext26t_32x4d()
    elif args.model == 'seresnext101_32x4d':
        net = seresnext101_32x4d()
    elif args.model == 'seresnext101_32x8d':
        net = seresnext101_32x8d()
    elif args.model == 'seresnext101d_32x8d':
        net = seresnext101d_32x8d()
    elif args.model == 'swinv2_large_window12to16_192to256_22kft1k':
        net = swinv2_large_window12to16_192to256_22kft1k()
    elif args.model == 'convnext_xlarge_in22ft1k':
        net = convnext_xlarge_in22ft1k()
    else:
        raise ValueError('Unknown model: {}'.format(args.model))

    return net, netT


def build_text_model(args, device, img_model):
    """
    Building text model(BERT) and return it.
    Also building combine model if args specified: "bilinear" and "concat"
    """
    text_model = None
    if args.bilinear_text or args.concat_text:
        text_model = BertClassifier()
        text_model.load_state_dict(torch.load("./best_bert.pth"))
        text_model = text_model.to(device)

        if args.bilinear_text:
            net = Combine_model(img_model, text_model, 'bilinear')
            net = net.to(device)
        elif args.concat_text:
            net = Combine_model(img_model, text_model, 'concat')
            net = net.to(device)
    else:
        net = img_model
    return text_model, net
