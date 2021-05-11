# model settings
model = dict(
    type='HybridTaskCascade', # Name of the detector
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=273,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=273,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=273,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=273,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=273,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=273,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
classes = [
    'water', 'pear', 'egg', 'grapes', 'butter', 'bread-white', 'jam',
    'bread-whole-wheat', 'apple', 'tea-green', 'white-coffee-with-caffeine',
    'tea-black', 'mixed-salad-chopped-without-sauce', 'cheese', 'tomato-sauce',
    'pasta-spaghetti', 'carrot', 'onion', 'beef-cut-into-stripes-only-meat',
    'rice-noodles-vermicelli', 'salad-leaf-salad-green', 'bread-grain',
    'espresso-with-caffeine', 'banana', 'mixed-vegetables', 'bread-wholemeal',
    'savoury-puff-pastry', 'wine-white', 'dried-meat', 'fresh-cheese',
    'red-radish', 'hard-cheese', 'ham-raw', 'bread-fruit',
    'oil-vinegar-salad-dressing', 'tomato', 'cauliflower', 'potato-gnocchi',
    'wine-red', 'sauce-cream', 'pasta-linguini-parpadelle-tagliatelle',
    'french-beans', 'almonds', 'dark-chocolate', 'mandarine',
    'semi-hard-cheese', 'croissant', 'sushi', 'berries', 'biscuits',
    'thickened-cream-35', 'corn', 'celeriac', 'alfa-sprouts', 'chickpeas',
    'leaf-spinach', 'rice', 'chocolate-cookies', 'pineapple', 'tart',
    'coffee-with-caffeine', 'focaccia', 'pizza-with-vegetables-baked',
    'soup-vegetable', 'bread-toast', 'potatoes-steamed', 'spaetzle',
    'frying-sausage', 'lasagne-meat-prepared', 'boisson-au-glucose-50g',
    'ma1-4esli', 'peanut-butter', 'chips-french-fries', 'mushroom',
    'ratatouille', 'veggie-burger', 'country-fries',
    'yaourt-yahourt-yogourt-ou-yoghourt-natural', 'hummus', 'fish', 'beer',
    'peanut', 'pizza-margherita-baked', 'pickle', 'ham-cooked',
    'cake-chocolate', 'bread-french-white-flour', 'sauce-mushroom',
    'rice-basmati', 'soup-of-lentils-dahl-dhal', 'pumpkin', 'witloof-chicory',
    'vegetable-au-gratin-baked', 'balsamic-salad-dressing', 'pasta-penne',
    'tea-peppermint', 'soup-pumpkin',
    'quiche-with-cheese-baked-with-puff-pastry', 'mango',
    'green-bean-steamed-without-addition-of-salt', 'cucumber',
    'bread-half-white', 'pasta', 'beef-filet', 'pasta-twist',
    'pasta-wholemeal', 'walnut', 'soft-cheese', 'salmon-smoked',
    'sweet-pepper', 'sauce-soya', 'chicken-breast', 'rice-whole-grain',
    'bread-nut', 'green-olives',
    'roll-of-half-white-or-white-flour-with-large-void', 'parmesan',
    'cappuccino', 'flakes-oat', 'mayonnaise', 'chicken', 'cheese-for-raclette',
    'orange', 'goat-cheese-soft', 'tuna', 'tomme', 'apple-pie', 'rosti',
    'broccoli', 'beans-kidney', 'white-cabbage', 'ketchup',
    'salt-cake-vegetables-filled', 'pistachio', 'feta', 'salmon', 'avocado',
    'sauce-pesto', 'salad-rocket', 'pizza-with-ham-baked', 'gruya-re',
    'ristretto-with-caffeine', 'risotto-without-cheese-cooked',
    'crunch-ma1-4esli', 'braided-white-loaf', 'peas',
    'chicken-curry-cream-coconut-milk-curry-spices-paste', 'bolognaise-sauce',
    'bacon-frying', 'salami', 'lentils', 'mushrooms',
    'mashed-potatoes-prepared-with-full-fat-milk-with-butter', 'fennel',
    'chocolate-mousse', 'corn-crisps', 'sweet-potato',
    'bircherma1-4esli-prepared-no-sugar-added',
    'beetroot-steamed-without-addition-of-salt', 'sauce-savoury', 'leek',
    'milk', 'tea', 'fruit-salad', 'bread-rye', 'salad-lambs-ear',
    'potatoes-au-gratin-dauphinois-prepared', 'red-cabbage', 'praline',
    'bread-black', 'black-olives', 'mozzarella', 'bacon-cooking',
    'pomegranate', 'hamburger-bread-meat-ketchup', 'curry-vegetarian', 'honey',
    'juice-orange', 'cookies', 'mixed-nuts', 'breadcrumbs-unspiced',
    'chicken-leg', 'raspberries', 'beef-sirloin-steak', 'salad-dressing',
    'shrimp-prawn-large', 'sour-cream', 'greek-salad', 'sauce-roast',
    'zucchini', 'greek-yaourt-yahourt-yogourt-ou-yoghourt', 'cashew-nut',
    'meat-terrine-pata-c', 'chicken-cut-into-stripes-only-meat', 'couscous',
    'bread-wholemeal-toast', 'craape-plain', 'bread-5-grain', 'tofu',
    'water-mineral', 'ham-croissant', 'juice-apple', 'falafel-balls',
    'egg-scrambled-prepared', 'brioche', 'bread-pita', 'pasta-haprnli',
    'blue-mould-cheese', 'vegetable-mix-peas-and-carrots', 'quinoa', 'crisps',
    'beef', 'butter-spread-puree-almond', 'beef-minced-only-meat',
    'hazelnut-chocolate-spread-nutella-ovomaltine-caotina', 'chocolate',
    'nectarine', 'ice-tea', 'applesauce-unsweetened-canned',
    'syrup-diluted-ready-to-drink', 'sugar-melon', 'bread-sourdough',
    'rusk-wholemeal', 'gluten-free-bread', 'shrimp-prawn-small',
    'french-salad-dressing', 'pancakes', 'milk-chocolate', 'pork',
    'dairy-ice-cream', 'guacamole', 'sausage', 'herbal-tea', 'fruit-coulis',
    'water-with-lemon-juice', 'brownie', 'lemon', 'veal-sausage', 'dates',
    'roll-with-pieces-of-chocolate', 'taboula-c-prepared-with-couscous',
    'croissant-with-chocolate-filling', 'eggplant', 'sesame-seeds',
    'cottage-cheese', 'fruit-tart', 'cream-cheese', 'tea-verveine', 'tiramisu',
    'grits-polenta-maize-flour', 'pasta-noodles', 'artichoke', 'blueberries',
    'mixed-seeds', 'caprese-salad-tomato-mozzarella', 'omelette-plain',
    'hazelnut', 'kiwi', 'dried-raisins', 'kolhrabi', 'plums', 'beetroot-raw',
    'cream', 'fajita-bread-only', 'apricots', 'kefir-drink', 'bread',
    'strawberries', 'wine-rosa-c', 'watermelon-fresh', 'green-asparagus',
    'white-asparagus', 'peach'
]

# DATA TRAIN AND TEST PIPELINE for data loading, data pre-processing (e.g data augmentation) and formatting operations
# img_norm_cfg: It's determined by the pretrained model, and you can just keep it the same if you use the model from pytorch model zoo (REF: https://github.com/open-mmlab/mmdetection/issues/354)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# Data augmetation using albumentations library
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='train/fixed_annotations_bbox.json',
        img_prefix='train/images/',
        pipeline=train_pipeline,
        data_root='./data/',
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file='val/fixed_annotations_bbox.json',
        img_prefix='val/images/',
        pipeline=test_pipeline,
        data_root='./data/',
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file='val/fixed_annotations_bbox.json',
        img_prefix='val/images/',
        pipeline=test_pipeline,
        data_root='./data/',
        classes=classes))

# evaluation config
evaluation = dict(
    metric=['bbox', 'segm'],  # Options are 'bbox', 'segm', 'proposal', 'proposal_fast'.
    interval=20)  # interval is the number of iterations between each evaluation
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy  (for more information see mask_rcnn_r50.py model)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[17, 24])

# epochs configs
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Checkpoint config to save partial results during training
checkpoint_config = dict(
    interval=1,  # number of epochs
    by_epoch=True, # wheter interval represents epochs or iterations
    save_optimizer=True,  # save optimizer is needed to resume experiments
    out_dir='/content/drive/My Drive/ML/models/htc_r50',  # the directory in which save checkpoints
    save_last=True)

# log configs
log_config = dict(
    interval=20,  # number of iterations
    hooks=[dict(type='TensorboardLoggerHook'),  # log used for tensorboard
           dict(type='TextLoggerHook')])  # text logs
log_level = 'INFO'

custom_hooks = [dict(type='NumClassCheckHook')]  # chekc if the number of classes in the head is consistent with those in the dataset
dist_params = dict(backend='nccl')

# runtime settings
load_from = '/content/drive/My Drive/ML/models/htc_r50/epoch_10.pth'   # used to load the model
resume_from = None  # used to resume and continue an experiment ( resume also the optimizer )
workflow = [('train', 1)]

# working dir in with the logs are saved and also the checkpoints if another folder is not specified in checkpoint_config
work_dir = '/content/drive/MyDrive/ML/models/htc_r50/logs'
# Set seed thus the results are more reproducible
seed = 0
gpu_ids = range(0, 1)