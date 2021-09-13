cfg_slimNet = {
    # 'anchors': [[15, 30], [48, 96], [128, 192], [256, 480]],
    'anchors': [[10, 20], [40, 64], [96, 128], [256, 320]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'loc_weight': 2.0,
    'clip': False,
    'batch_size': 8,
    'epoch': 250,
    'image_size': 448,
    'out_channels': [16, 24, 32, 64, 96, 128],
    'fpn_in_list': [32, 64, 96, 128],
    'fpn_out_list': [32, 64, 96, 128],
    'ssh_out_channel': 64,
    'num_classes': 2
}

cfg_slimNet2 = {
    'anchors': [[10, 20, 40], [64, 96], [128, 192], [256, 450, 640]],
    # 'anchors': [[10, 20], [40, 64], [96, 128], [256, 320]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'loc_weight': 2.0,
    'clip': False,
    'batch_size': 64,
    'epoch': 251,
    'image_size': 832,
    'out_channels': [16, 32, 48, 64, 96, 128],
    'fpn_in_list': [48, 64, 96, 128],
    'fpn_out_list': [48, 64, 96, 128],
    'ssh_out_channel': 64,
    'num_classes': 2
}

cfg_slimNet3 = {
    'anchors': [[10, 20, 40], [64, 96], [128, 192], [256, 384, 512]],
    # 'anchors': [[10, 20], [40, 64], [96, 128], [256, 320]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'loc_weight': 2.0,
    'clip': False,
    'batch_size': 20,
    'epoch': 251,
    'image_size': 640,
    'out_channels': [16, 32, 48, 64, 96, 128],
    'fpn_in_list': [48, 64, 96, 128],
    'fpn_out_list': [48, 64, 96, 128],
    'ssh_out_channel': 64,
    'num_classes': 2
}