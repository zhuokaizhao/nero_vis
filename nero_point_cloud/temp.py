import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

result_dir = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/nero_point_cloud/output/'
result_names = [
    'point_transformer_model_rot_False_e_20.npz',
    'point_transformer_model_rot_True_e_20.npz',
]
result = np.load(os.path.join(result_dir, result_names[0]))

r = list(result['all_axis'])
theta = list(result['all_angles'])
accuracy = list(result['instance_accuracies'])
df = pd.DataFrame(list(zip(r, theta, accuracy)), columns =['r', 'theta', 'accuracy'])

ntheta = 30
dtheta = 360/ntheta
nradius = 20
dradius = max(r)/nradius

colors = ['#000052', '#0c44ac', '#faf0ca', '#ed0101', '#970005']

cm = LinearSegmentedColormap.from_list('custom', colors, N=10)
cm.set_bad(color='white')

patches = []
avg_temp = []

for nr in range(nradius, 0, -1):  # Outside to Inside
    start_r = (nr-1)*dradius
    end_r = (nr)*dradius
    for nt in range(0,ntheta):
        start_t = nt*dtheta
        end_t = (nt+1)*dtheta

        stripped = df[(df['r']>=start_r) & (df['r']<end_r) &
            (df['theta']>=start_t) & (df['theta']<end_t)]

        avg_temp.append(stripped['accuracy'].mean())
        wedge = mpatches.Wedge(0,end_r, start_t, end_t)
        patches.append(wedge)


collection = PatchCollection(
    patches,linewidth=0.0,
    edgecolor=['#000000' for x in avg_temp],
    facecolor=cm([(x-263.15 )/(303.15 -263.15 ) for x in avg_temp])
)

fig = plt.figure(
    figsize=(40,20),
    dpi=200,
    edgecolor='w',
    facecolor='w'
)
ax = fig.add_subplot()
ax.add_collection(collection)
# Clean up the image canvas and save!
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

# plt.savefig('toronto.png')
plt.show()

