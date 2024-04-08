import numpy as np
import os
import ntpath
import time
import visdom
from . import util
from . import html

class Visualizer():
    def __init__(self, args):
        self.use_html = False
        self.display_id = args.display_id
        self.win_size = args.display_winsize
        self.name = args.output_dir.split('/')[-1]

        if self.display_id > 0:
            self.vis = visdom.Visdom(server=args.display_server, port=args.display_port)
            self.ncols = args.display_ncols

        if self.use_html:
            self.web_dir = os.path.join(args.output_dir, self.name, 'web')
            self.img_train_dir = os.path.join(self.web_dir, 'train_images')
            self.img_test_dir = os.path.join(self.web_dir, 'test_images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_train_dir])
            util.mkdirs([self.web_dir, self.img_test_dir])


    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, train=True):
        if self.display_id > 0: # show images in the browser
            if self.ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                '''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                '''
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1
        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if train:
                    img_path = os.path.join(self.img_train_dir, 'epoch%.3d_%s.png' % (epoch, label))
                else:
                    img_path = os.path.join(self.img_test_dir, '%d.png' % (epoch))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, errors):
        if self.display_id > 0:
            if not hasattr(self, 'plot_data_train'):
                self.plot_data_train = {'X': [], 'Y': [], 'legend': list(errors.keys())}
            self.plot_data_train['X'].append(epoch + counter_ratio)
            self.plot_data_train['Y'].append([errors[k] for k in self.plot_data_train['legend']])
            self.vis.line(
                X=np.stack([np.array(self.plot_data_train['X'])] * len(self.plot_data_train['legend']), 1),
                Y=np.array(self.plot_data_train['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data_train['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)


    def plot_test_errors(self, epoch, counter_ratio, errors):
        if self.display_id > 0:
            if not hasattr(self, 'plot_data'):
                self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
            self.plot_data['X'].append(epoch + counter_ratio)
            self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id + 10)


    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
