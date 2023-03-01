"""
Author: Benny
Date: Nov 2019
"""
from dataset import ModelNetDataLoader
import numpy as np
import os
import torch
import time
import logging
from tqdm import tqdm
import provider
import importlib
import shutil
import hydra
import omegaconf

import quaternions


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for _, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


@hydra.main(config_path='config', config_name='cls', version_base=None)
def main(args):
    # allow new entries to be added
    omegaconf.OmegaConf.set_struct(args, False)

    # parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    logger.info(f'\nParameters:')
    for key in args.keys():
        logger.info(f'{key}: {args[key]}')
    logger.info('\n')

    # load data
    logger.info('Loading dataset')
    DATA_PATH = hydra.utils.to_absolute_path(args['data_dir'])
    if args['mode'] == 'train':
        TRAIN_DATASET = ModelNetDataLoader(
            root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal
        )
        TEST_DATASET = ModelNetDataLoader(
            root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal
        )
        train_data_oader = torch.utils.data.DataLoader(
            TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        test_data_loader = torch.utils.data.DataLoader(
            TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    elif args['mode'] == 'test':
        TEST_DATASET = ModelNetDataLoader(
            root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal
        )
        testDataLoader = torch.utils.data.DataLoader(
            TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4
        )


    # load model
    args.num_class = 40
    if args.normal:
        args.input_dim = 6
    else:
        args.input_dim = 3
    # get code ready for model
    shutil.copy(hydra.utils.to_absolute_path(f'models/{args.model.name}/model.py'), '.')

    model = getattr(
        importlib.import_module(f'models.{args.model.name}.model'),
        'PointTransformerCls',
    )(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if args['mode'] == 'train':
        # try loading the checkpoint model
        try:
            checkpoint = torch.load(f'{args.model.checkpoint_path}')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'Continue training with checkpoint model {args.model.checkpoint_path}')
        except:
            logger.info('No existing model, starting training from scratch')
            start_epoch = 0

        # do the training
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )

        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
        cur_session_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0
        best_epoch = 0
        mean_correct = []

        # TRANING
        logger.info('Training starts')
        for epoch in range(start_epoch, args.epoch):
            logger.info(f'Epoch {cur_session_epoch+1} ({epoch+1}/{args.epoch}):')
            epoch_start = time.time()

            # turn on train before each epoch starts
            model.train()
            for _, data in tqdm(
                enumerate(train_data_oader, 0), total=len(train_data_oader), smoothing=0.9
            ):
                points, target = data
                points = points.data.numpy()
                # doing data augmentation
                # randomly replace a set of points with the first point
                points = provider.random_point_dropout(points)
                # randomly scale the points from 0.8 to 1.25
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
                # Randomly shift point cloud
                points[:, :, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
                # if requested, random rotate given percentage of 180 degrees
                if args.random_rotate:
                    # randomly pick a rotation axis and angle
                    # each batch of points shares the same rotation axis and angle
                    rot_axis = quaternions.pick_random_axis(len(points))
                    rot_degrees = np.random.uniform(-180, 180, len(points))
                    # rotate the points
                    points[:,:, 0:3] = quaternions.rotate(points[:, :, 0:3], rot_axis, rot_degrees)

                # convert points data to tensor
                points = torch.Tensor(points)
                target = target[:, 0]
                # move to GPU
                points, target = points.cuda(), target.cuda()

                # clear optimizer
                optimizer.zero_grad()

                # train the model
                pred = model(points)
                loss = criterion(pred, target.long())
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))
                loss.backward()
                optimizer.step()
                global_step += 1

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            logger.info(f'Epoch {epoch+1} completed in {epoch_time} seconds.')

            scheduler.step()

            train_instance_acc = np.mean(mean_correct)
            logger.info(f'Train Instance Accuracy: {train_instance_acc}')

            # validate and save model every 5 epochs
            if (epoch + 1) % args.checkpoint_freq == 0:
                with torch.no_grad():
                    instance_acc, class_acc = test(model.eval(), test_data_loader)

                    # use instance accuracy to determine best epoch
                    if (instance_acc >= best_instance_acc):
                        best_instance_acc = instance_acc
                        best_epoch = epoch + 1
                        logger.info(f'Best epoch is {best_epoch}')

                    if (class_acc >= best_class_acc):
                        best_class_acc = class_acc

                    logger.info(
                        f'Test Instance Accuracy: {instance_acc}, Class Accuracy: {class_acc}'
                    )
                    logger.info(
                        f'Best Instance Accuracy: {best_instance_acc}, Class Accuracy: {best_class_acc}'
                    )

                    # save model
                    save_path = os.path.join(
                        args.model_dir,
                        f'{args.model.name}_model_rot_{args.random_rotate}_e_{epoch+1}.pth'
                    )
                    state = {
                        'epoch': epoch + 1,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, save_path)
                    logger.info(f'Checkpoint model saved to {save_path}')
                    cur_session_epoch += 1

        logger.info('End of training')

    # test mode
    elif args['mode'] == 'test':
        # loading the checkpoint model is required
        checkpoint = torch.load(f'{args.model.checkpoint_path}')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Testing with checkpoint model {args.model.checkpoint_path}')




if __name__ == '__main__':
    main()