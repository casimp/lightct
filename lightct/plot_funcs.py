import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np


def recentre_plot(image, cor_offset):
    fig, ax_array = plt.subplots(1, 2, figsize=(10, 6))
    fig.canvas.set_window_title('Recentred Projection')
    height, width = image.shape
    if cor_offset <= 0:
        poly_pnts = [[width + cor_offset, 0], [width, 0],
                     [width, height], [width + cor_offset, height]]
    else:
        poly_pnts = [[0, 0], [cor_offset, 0],
                     [cor_offset, height], [0, height]]

    ax_array[0].imshow(image)
    centre = width / 2 + cor_offset / 2
    ax_array[0].plot([centre, centre], [0, height], 'k-',
                     linewidth=2, label='New COR')
    ax_array[0].plot([width / 2, width / 2],
                     [0, height], 'r-', linewidth=2, label='Old COR')
    ax_array[0].legend()
    ax_array[0].set_xlim([0, image.shape[1]])
    ax_array[0].set_ylim([image.shape[0], 0])

    if cor_offset >= 0:
        offset_image = np.copy(image[:, cor_offset:])
    else:
        offset_image = np.copy(image[:, :cor_offset])

    ax_array[1].imshow(offset_image)
    ax_array[1].plot([offset_image.shape[1] / 2, offset_image.shape[1] / 2],
                     [0, height], 'k-', linewidth=2, label='New COR')
    ax_array[1].legend()

    ax_array[1].set_xlim([0, offset_image.shape[1]])
    ax_array[1].set_ylim([offset_image.shape[0], 0])

    ax_array[0].add_patch(patches.Polygon(poly_pnts, closed=True,
                                          fill=False, hatch='///', color='k'))
    ax_array[0].set_title('Uncropped')
    ax_array[1].set_title('Cropped and centred')

    plt.show()


class SetAngleInteract(object):

    def __init__(self, im_stack, p0):
        self.im_stack = im_stack
        self.p0 = p0
        self.num_images = None
        self.angles = None

    def interact(self):
        backend = matplotlib.get_backend()
        err = ("Matplotlib running inline. Plot interaction not possible."
               "\nTry running %matplotlib in the ipython console (and "
               "%matplotlib inline to return to default behaviour). In "
               "standard console use matplotlib.use('TkAgg') to interact.")

        assert backend != 'module://ipykernel.pylab.backend_inline', err
        fig, ax_array = plt.subplots(1, 2, figsize=(10, 5))

        ax_slider = plt.axes([0.2, 0.07, 0.5, 0.05])
        ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])

        for ax in ax_array:
            ax.imshow(self.im_stack[:, :, self.p0])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.subplots_adjust(bottom=0.2)
        nfiles = self.im_stack.shape[-1] + 1
        window_slider = Slider(ax_slider, 'Image', self.p0, nfiles, valinit=0)
        store_button = Button(ax_button, r'Save - 360')

        def slider_update(val):
            ax_array[1].imshow(self.im_stack[:, :, int(window_slider.val)])
            window_slider.valtext.set_text('%i' % window_slider.val)
            fig.canvas.draw_idle()

        window_slider.on_changed(slider_update)

        def store_data(label):
            # Check this is correct - proj_ref!!!
            self.num_images = int(window_slider.val) - self.p0 + 1
            self.angles = np.linspace(0, 360, self.num_images)
            plt.close()

        store_button.on_clicked(store_data)
        plt.show()
        return window_slider, store_button
