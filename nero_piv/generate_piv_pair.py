# The script generates the pair data for PIV
import os
import math
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

import plot

# wrap around the particles
def wrap_around(in_value, min, max):

    while (in_value < min or in_value > max):
        if (in_value < min):
            in_value += (max-min)
        if (in_value > max):
            in_value -= (max-min)

    return in_value


def write_flo(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    # flow = flow[0, :, :, :]
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()


# assign intensity from distance and peak value
def get_intensity(center_pos, pos, peak_intensity, diameter):

    inside_exp_top = -(pos[0] - center_pos[0])**2 - (pos[1] - center_pos[1])**2
    inside_exp_bottom = 1 / 8 * diameter**2
    intensity = peak_intensity * np.exp(inside_exp_top/inside_exp_bottom)

    return intensity


# form image from particle position array
def form_image(positions, diameters, peak_intensities, image_size):

    image = np.zeros(image_size)

    for i in range(len(positions)):
        pos = positions[i]
        if pos.any() < 0 or pos.any() >= image_size[0]:
            continue

        # draw peak intensity as the center pixel
        image[pos[0], pos[1]] = peak_intensities[i]

        # if diameter larger than zero, draw other pixels
        if diameters[i] > 1:
            if diameters[i] == 2:
                for j in range(-1, 1):
                    for k in range(0, 2):
                        if j == 0 and k == 0:
                            continue
                        elif pos[0]+j >= 0 and pos[0] + j < image_size[0] and pos[1]+k >= 0 and pos[1] + k < image_size[1]:
                            image[pos[0]+j, pos[1]+k] = get_intensity(pos, (pos[0]+j, pos[1]+k), peak_intensities[i], diameters[i])

            elif diameters[i] == 3:
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if j == 0 and k == 0:
                            continue
                        elif pos[0]+j >= 0 and pos[0] + j < image_size[0] and pos[1]+k >= 0 and pos[1] + k < image_size[1]:
                            image[pos[0]+j, pos[1]+k] = get_intensity(pos, (pos[0]+j, pos[1]+k), peak_intensities[i], diameters[i])

            elif diameters[i] == 4:
                for j in range(-2, 2):
                    for k in range(-1, 3):
                        if j == 0 and k == 0:
                            continue
                        elif pos[0]+j >= 0 and pos[0] + j < image_size[0] and pos[1]+k >= 0 and pos[1] + k < image_size[1]:
                            image[pos[0]+j, pos[1]+k] = get_intensity(pos, (pos[0]+j, pos[1]+k), peak_intensities[i], diameters[i])
            else:
                raise Exception(f'Unsupported diameter {diameters[i]}')

    return image


# save image
def save_image(image, image_size, path):

    image = image.astype(np.uint8).reshape((image_size[0], image_size[1]))
    image = Image.fromarray(image)
    image.save(path)


# save labels to flow
def save_velocity(mode, velocity, size, path):

    # when velocity size is larger than output size, take the center
    if size != velocity.shape[:2]:
        center_start_idx = (velocity.shape[0] - size[0])//2
        center_end_idx = (velocity.shape[0] + size[0])//2
        velocity = velocity[center_start_idx:center_end_idx, center_start_idx:center_end_idx, :]

    print(velocity.shape)
    exit()

    if mode == 'npy':
        np.save(path, velocity)
    elif mode == 'flo':
        write_flo(path, velocity)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='NERO plots with MNIST')
    # mode (train, test or analyze)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # input h5 data dir
    parser.add_argument('-i', '--input_dir', required=True, action='store', nargs=1, dest='input_dir')
    # name of the data
    parser.add_argument('--data_name', required=True, action='store', nargs=1, dest='data_name')
    # image dimension
    parser.add_argument('-s', '--image_size', action='store', nargs=1, dest='image_size')
    # output figs directory
    parser.add_argument('-o', '--output_dir', required=True, action='store', nargs=1, dest='output_dir')
    # velocity field save type
    parser.add_argument('-t', '--output_type', required=True, action='store', nargs=1, dest='output_type')
    # if visualizing data
    parser.add_argument('--vis', action='store_true', dest='vis', default=False)
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)

    args = parser.parse_args()

    # input variables
    mode = args.mode[0]
    input_dir = args.input_dir[0]
    data_name = args.data_name[0]
    if args.image_size != None:
        image_size = (int(args.image_size[0].split('x')[0]), int(args.image_size[0].split('x')[1]))
    else:
        image_size = (256, 256)
    # output graph directory
    output_dir = args.output_dir[0]
    output_type = args.output_type[0]
    vis_data = args.vis
    verbose = args.verbose

    if verbose:
        print(f'\nMode: {mode}')
        print(f'Input dir: {input_dir}')
        print(f'Data name: {data_name}')
        print(f'Output image size: {image_size}')
        print(f'Output directory: {output_dir}\n')

    # every 10 DNS time-steps (0.0002 per step) is stored, data downloaded has stride 20
    time_stride = 20
    dt = 0.002 * time_stride

    # load the input h5
    loaded_file = h5py.File(input_dir, 'r')

    # potential rotation angles
    if mode == 'train' or mode == 'val':
        rotation_angles = [0]
    elif mode == 'test':
        rotation_angles = list(range(0, 360, 45))

    # when doing rotation, simulation/image field of view needs to be larger in the first place
    if len(rotation_angles) == 1:
        simulation_size = image_size
    else:
        simulation_size = np.ceil(image_size / np.sqrt(2) * 2).astype(int)

    # training takes all the z slices, while testing only takes one slice
    if mode == 'train':
        z_range = list(range(0, 101))
    else:
        # for test, only the center z slice is used
        z_range = [51]

    # all rotations use the same set of particle initializations
    # however, different z slice and time step should have different initializations
    all_timesteps = list(range(1, 5024, 20))
    all_timesteps = list(range(1, 4, 20))

    # velocity goes from time step 1 to 5024 with stride 20
    for t in tqdm(all_timesteps):
        # load the velocity
        cur_key = f'Velocity_{str(t).zfill(4)}'
        cur_velocity = loaded_file[cur_key]

        # loaded data has coordinate (z, y, x, (v_z, v_y, v_x))
        # convert to (x, y, z, (v_x, v_y, v_z))
        # in image axis row is y, col is x, thus [0, 1, 2] to [2, 0, 1], instead of [2, 1, 0]
        cur_velocity = np.moveaxis(cur_velocity, [0, 1, 2], [2, 0, 1])
        cur_velocity = cur_velocity[:, :, :, ::-1]

        # convert velocity from [0, 2pi] to [0, 1024]
        cur_velocity = cur_velocity / (2*np.pi) * 1024

        # for each z
        for z in enumerate(z_range):
            # only velocities in x and y
            cur_velocity_2d = cur_velocity[:, :, z, :2]

            # initalize current time step and z slice's randomization
            # particle density and number of particles
            particle_densities = np.random.uniform(0.05, 0.1, size=1)
            num_particles = int(particle_densities * simulation_size[0] * simulation_size[1])
            # random initalize particle positions
            frame1_particle_pos = np.random.randint(low=0, high=simulation_size[0], size=[num_particles, 2], dtype=int)
            # random initialize particle radius
            all_particle_diameters = np.random.randint(low=1, high=5, size=num_particles, dtype=int)
            # random initialize peak intensity
            all_peak_intensities = np.random.randint(low=200, high=256, size=num_particles, dtype=int)
            # particle positions in the second frame
            frame2_particle_pos = np.zeros(frame1_particle_pos.shape, dtype=int)
            # when doing rotations, we need to keep 0-rotation data untouched all the time
            if len(rotation_angles) != 1:
                frame1_rotated_particle_pos = np.zeros(frame1_particle_pos.shape, dtype=int)
                frame2_rotated_particle_pos = np.zeros(frame2_particle_pos.shape, dtype=int)

            # process with each rotation angle
            for rot in rotation_angles:
                cur_output_dir = os.path.join(output_dir, f'rotated_{rot}')
                os.makedirs(cur_output_dir, exist_ok=True)

                # rotate the velocity field if needed
                if rot != 0:
                    # rotate the entire loaded velocity field (counterclock wise)
                    cur_label_rotated_temp = rotate(cur_velocity_2d, angle=int(rot), reshape=False)
                    # rotate each velocity vector too (counterclock wise)
                    cur_velocity_2d[:, :, 0] = cur_label_rotated_temp[:, :, 0] * np.cos(math.radians(rot)) + cur_label_rotated_temp[:, :, 1] * np.sin(math.radians(rot))
                    cur_velocity_2d[:, :, 1] = -cur_label_rotated_temp[:, :, 0] * np.sin(math.radians(rot)) + cur_label_rotated_temp[:, :, 1] * np.cos(math.radians(rot))

                # take only the simulation size as velocity field
                center_start_idx = (cur_velocity_2d.shape[0] - simulation_size[0])//2
                center_end_idx = (cur_velocity_2d.shape[0] + simulation_size[0])//2
                cur_velocity_2d = cur_velocity_2d[center_start_idx:center_end_idx, center_start_idx:center_end_idx, :]

                # sanity check to make sure velocity fields have the same size as particle images
                if cur_velocity_2d.shape[:2] < image_size:
                    raise Exception(f'Velocity field shape {cur_velocity_2d.shape} smaller than image shape {image_size}')

                # only compute/simulate particles when 0 rotation, otherwise just rotate the particle locations
                if rot == 0:
                    # compute frame 2 particle positions
                    frame2_particle_pos[:, 1] = frame1_particle_pos[:, 1] + cur_velocity_2d[frame1_particle_pos[:, 0], frame1_particle_pos[:, 1], 1] * dt
                    frame2_particle_pos[:, 0] = frame1_particle_pos[:, 0] + cur_velocity_2d[frame1_particle_pos[:, 0], frame1_particle_pos[:, 1], 0] * dt

                    # form image for both frame 1 and 2
                    image_1 = form_image(frame1_particle_pos, all_particle_diameters, all_peak_intensities, image_size)
                    image_2 = form_image(frame2_particle_pos, all_particle_diameters, all_peak_intensities, image_size)

                # rotate the 0-rotation particle positions for other rotation angles
                else:
                    # first frame
                    frame1_rotated_particle_pos[:, 1] = -(frame1_particle_pos[:, 0]-128) * np.sin(math.radians(rot)) + (frame1_particle_pos[:, 1]-128) * np.cos(math.radians(rot)) + 128
                    frame1_rotated_particle_pos[:, 0] = (frame1_particle_pos[:, 0]-128) * np.cos(math.radians(rot)) + (frame1_particle_pos[:, 1]-128) * np.sin(math.radians(rot)) + 128
                    # second frame
                    frame1_rotated_particle_pos[:, 1] = -(frame2_particle_pos[:, 0]-128) * np.sin(math.radians(rot)) + (frame2_particle_pos[:, 1]-128) * np.cos(math.radians(rot)) + 128
                    frame1_rotated_particle_pos[:, 0] = (frame2_particle_pos[:, 0]-128) * np.cos(math.radians(rot)) + (frame2_particle_pos[:, 1]-128) * np.sin(math.radians(rot)) + 128

                    # form image for frame 1 and 2
                    image_1 = form_image(frame1_rotated_particle_pos, all_particle_diameters, all_peak_intensities, image_size)
                    image_2 = form_image(frame2_rotated_particle_pos, all_particle_diameters, all_peak_intensities, image_size)

                # save image 1 and 2
                image_1_path = os.path.join(cur_output_dir, f'isotropic_1024_image_{data_name}_z_{z}_t_{t}_0.png')
                save_image(image_1, image_size, image_1_path)
                image_2_path = os.path.join(cur_output_dir, f'isotropic_1024_image_{data_name}_z_{z}_t_{t}_1.png')
                save_image(image_2, image_size, image_2_path)

                # save the ground truth
                label_path = os.path.join(cur_output_dir, f'isotropic_1024_velocity_{data_name}_z_{z}_t_{t}.{output_type}')
                # Normally, since training is for LiteFlowNet-en, it takes flo format
                if mode == 'train' or mode == 'val':
                    if output_type == 'npy':
                        raise Exception(f'For training flo should be used as save velocity type')

                # only save the image size velocity
                save_velocity(output_type, cur_velocity_2d, image_size, label_path)

                # visualize ground truth velocity if requested
                if vis_data:
                    max_truth = 3
                    # visualize ground truth
                    flow_vis, _ = plot.visualize_flow(cur_velocity_2d)
                    # convert to Image
                    flow_vis_image = Image.fromarray(flow_vis)
                    # display the image
                    plt.imshow(flow_vis_image)
                    # superimpose quiver plot on color-coded images
                    skip = int(image_size[0]/256 * 7)
                    x = np.linspace(0, cur_velocity_2d.shape[0]-1, cur_velocity_2d.shape[0])
                    y = np.linspace(0, cur_velocity_2d.shape[1]-1, cur_velocity_2d.shape[1])
                    y_pos, x_pos = np.meshgrid(x, y)
                    Q = plt.quiver(y_pos[::skip, ::skip],
                                    x_pos[::skip, ::skip],
                                    cur_velocity_2d[::skip, ::skip, 0]/max_truth,
                                    -cur_velocity_2d[::skip, ::skip, 1]/max_truth,
                                    # scale=4.0,
                                    scale_units='inches')
                    Q._init()
                    assert isinstance(Q.scale, float)
                    plt.show()
                    exit()




if __name__ == '__main__':
    main()