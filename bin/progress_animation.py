import matplotlib as mpl
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt

mpl.rcParams['toolbar'] = 'None'

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(3.5,3.5))
fig.canvas.set_window_title('Acquisition')

patch = Wedge((.5, .5), .45, 90, 90, width=0.15)
ax.add_patch(patch)
ax.axis('equal')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.axis('off')
t = ax.text(0.5, 0.5, '0%%', fontsize=15, ha='center', va='center')
di = 1
for i in range(0,360,di):
    patch.set_theta1(90 - i - di)
    progress = 100 * (i+di) / 360
    t.set_text('%02d%%' % progress)
    plt.pause(0.0001)
    #plt.show()
plt.close()