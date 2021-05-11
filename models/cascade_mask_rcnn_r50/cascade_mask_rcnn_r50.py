model = dict(
    type='CascadeRCNN',
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
        type='CascadeRoIHead',
        num_stages=3,  # This is the number of stages in cascade
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
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=273,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
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
                    match_low_quality=False,
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
                    match_low_quality=False,
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
                    match_low_quality=False,
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
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
# Dataset configuration
dataset_type = 'CocoDataset'  # type of annotations
data_root = './data/'  # root path of data
# the 273 classes of the Food Recognition challenge
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
img_norm_cfg = dict(  # Image normalization config to normalize the input images
    mean=[123.675, 116.28, 103.53],  # Mean values used to pre-training the pre-trained backbone models
    std=[58.395, 57.12, 57.375],  # Standard variance used to pre-training the pre-trained backbone models
    to_rgb=True)
# Data augmetation using albumentations library
# ref: https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/datasets/pipelines/transforms.html
# ref: https://albumentations.ai/docs/
albumentations_train_transforms = [
    dict(
        type='ShiftScaleRotate',  # Randomly apply affine transforms: translate, scale and rotate the input.
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=20,
        p=0.7),  # probability of applying the transform
    # dict(type='RandomRotate90',
    #     p=0.5),
    # dict(type='CLAHE',
    #     p=0.15),
    dict(
        type='RandomBrightnessContrast',  # Randomly change brightness and contrast of the input image
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.3),
     dict(
        type="OneOf", # Select one of transforms to apply. Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights. Default probability is 0.5
        transforms=[
            dict(type="Blur", blur_limit=5),  # Blur the input image using a random-sized kernel
            dict(type="MotionBlur", blur_limit=5),
            dict(type="GaussNoise", var_limit=25),
            dict(type="ImageCompression", quality_lower=75),  # decrease jpeg with lower bound on the image quality
        ],
        p=0.5,
    ),
    #  dict(
    #     type="CoarseDropout",  # Augmenter that sets rectangular areas within images to zero.
    #     max_holes=30,
    #     max_height=30,
    #     max_width=30,
    #     min_holes=5,
    #     min_height=10,
    #     min_width=10,
    #     fill_value=img_norm_cfg["mean"][::-1],
    #     p=0.1,  # low probability
    # ),
    #dict(type='JpegCompression', quality_lower=75, quality_upper=99, p=0.25),
    #dict(type="GaussNoise",p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    #dict(
    #    type='OneOf',
    #    transforms=[
    #        dict(type='Blur', blur_limit=3, p=1.0),
    #        dict(type='GaussianBlur', blur_limit=3, p=1.0)
    #    ],
    #    p=0.15),
]

# TRAINING PIPELINE
train_pipeline = [
  # A. DATA LOADING
    dict(type='LoadImageFromFile'),  # load images from file path
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # load annotations for current image

  # B. DATA PRE-PROCESSING
    dict(type='Resize',  # Augmentation pipeline that resize the images and their annotations
         img_scale=[(480, 480), (960, 960)],   # multiscale training -> resizing images to different scales at each iteration
         # ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly sample a scale from the multiscale range.
         ratio_range=None,
         multiscale_mode='range',
         keep_ratio=True),  # keep the aspect ratio when resizing the image.
    dict(type='Albu',
         transforms=albumentations_train_transforms,  # A list of albu transformations
         # bbox_params and keymap are needed to use albumentations with bbox and masks
         bbox_params=dict(  # Parameters of bounding boxes for the albumentation `Compose`
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0.0,
             filter_lost_elements=True),
         keymap={  # Contains {'input key':'albumentation-style key'}
             'img': 'image',
             'gt_masks': 'masks',
             'gt_bboxes': 'bboxes'
         },
         update_pad_shape=False,
         skip_img_without_anno=False),
    dict(type='RandomFlip',  #  Augmentation pipeline that flip the images and their annotations
         # ``flip_ratio`` is float, ``direction`` is string: the image will be ``direction``ly flipped with probability of ``flip_ratio`` .
         flip_ratio=0.5,  # The ratio or probability to flip
         direction='horizontal'),  # valid directions ['horizontal', 'vertical', 'diagonal']
    dict(type='Normalize', **img_norm_cfg),  # Augmentation pipeline that normalize the input images
    dict(type='Pad', size_divisor=32),  # Pad config, with the number the padded images should be divisible

  # C. FORMATTING
    dict(type='DefaultFormatBundle'),  # It simplifies the pipeline of formatting common fields, including "img", "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".Formatted using: transpose, to tensor, to DataContainer
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])  # Pipeline that decides which keys in the data should be passed to the detector
]
test_pipeline = [
  # A. DATA LOADING
    dict(type='LoadImageFromFile'), # load image from file

  # B. TEST TIME AUGMENTATION
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),  # Decides the largest scale for testing, used for the Resize pipeline
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
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
# optimizer config
optimizer = dict(
    type='SGD',
    lr=0.0025,  # the default learning rate is 0.2 and is set for 8-GPU, we divide it by 8 since we use a single GPU 0.2/8
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
'''
Warm-up is a way to reduce the primacy effect of the early training examples.
The learning rate is increased linearly over the warm-up period. 
If the target learning rate is p and the warm-up period is n, 
 the first batch iteration uses 1*p/n for its learning rate; the second uses 2*p/n, and so on: 
 iteration i uses i*p/n, until we hit the nominal rate at iteration n.
This means that the first iteration gets only 1/n of the primacy effect.
'''
lr_config = dict(
    by_epoch=True,  # LR changes epoch by epoch
    policy='step',   # The policy of scheduler, also support CosineAnnealing, Cyclic, etc.
    warmup='linear',  # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500,  # The number of iterations for warmup
    warmup_ratio=0.001,  # The ratio of the starting learning rate used for warmup
    gamma=0.1,  # learning rate decay
    step=[16, 19])  # Steps to decay the learning rate

# epochs configs
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Checkpoint config to save partial results during training
checkpoint_config = dict(
    interval=1,  # number of epochs
    by_epoch=True, # wheter interval represents epochs or iterations
    save_optimizer=True,  # save optimizer is needed to resume experiments
    out_dir='/content/drive/My Drive/ML/models/cascade_mask_rcnn_r50',  # the directory in which save checkpoints
    save_last=True)

# log configs
log_config = dict(
    interval=20,  # number of iterations
    hooks=[dict(type='TensorboardLoggerHook'),  # log used for tensorboard
           dict(type='TextLoggerHook')])  # text logs
log_level = 'INFO'

custom_hooks = [dict(type='NumClassCheckHook')]  # chekc if the number of classes in the head is consistent with those in the dataset
dist_params = dict(backend='nccl')

load_from = '/content/drive/My Drive/ML/models/cascade_mask_rcnn_r50/epoch_10.pth'   # used to load the model
resume_from = None  # used to resume and continue an experiment ( resume also the optimizer )
workflow = [('train', 1)]

# working dir in with the logs are saved and also the checkpoints if another folder is not specified in checkpoint_config
work_dir = '/content/drive/My Drive/ML/models/cascade_mask_rcnn_r50/logs'
# Set seed thus the results are more reproducible
seed = 0
gpu_ids = range(0, 1)