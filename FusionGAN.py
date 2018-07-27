from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

class FusionGAN(object) :
    def __init__(self, sess, args):
        self.model_name = 'FusionGAN'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.L1_weight = args.L1_weight

        """ Generator """


        """ Discriminator """
        self.sn = args.sn

        self.img_size = args.img_size
        self.img_ch = args.img_ch


        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))

        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")

        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_identity, x_shape, reuse=False, scope="generator"):
        channel = self.ch
        """
        In paper, 
        x_identity = x
        x_shape = y
        """
        with tf.variable_scope(scope, reuse=reuse):
            identity_list = []
            shape_list = []

            x_identity = conv(x_identity, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='i_conv')
            x_identity = batch_norm(x_identity, scope='i_batch_norm')
            x_identity = relu(x_identity)

            x_shape = conv(x_shape, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='s_conv')
            x_shape = batch_norm(x_shape, scope='s_batch_norm')
            x_shape = relu(x_shape)

            for i in range(2) :
                x_identity = conv(x_identity, channel * 2, kernel=3, stride=2, pad=1, scope='i_conv_' + str(i))
                x_identity = batch_norm(x_identity, scope='i_batch_norm_' + str(i))
                x_identity = relu(x_identity)

                x_shape = conv(x_shape, channel * 2, kernel=3, stride=2, pad=1, scope='s_conv_' + str(i))
                x_shape = batch_norm(x_shape, scope='s_batch_norm_' + str(i))
                x_shape = relu(x_shape)

                identity_list.append(x_identity)
                shape_list.append(x_shape)

                channel = channel * 2

            for i in range(2) :
                x_identity = resblock(x_identity, channel, scope='i_resblock_' + str(i))
                x_shape = resblock(x_shape, channel, scope='s_resblock_' + str(i))

            x = concat([x_identity, x_shape])

            x = shortcut_resblock(x, channel, scope='shortcut_resblock')
            x = resblock(x, channel, scope='resblock')
            x = concat([identity_list[1], shape_list[1], x])

            for i in range(2) :
                x = deconv(x, channel//2, kernel=3, stride=2, scope='deconv_' + str(i))
                x = batch_norm(x, scope='deconv_batch_norm_' + str(i))
                x = relu(x)

                if i == 0:
                    x = concat([identity_list[0], shape_list[0], x])

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x
    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_identity, x, reuse=False, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = concat([x_identity, x])

            x = conv(x, channel, kernel=4, stride=2, pad=1, sn=self.sn, scope='conv')
            x = lrelu(x, 0.2)

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, sn=self.sn, scope='conv_s2_' + str(i))
                x = batch_norm(x, scope='batch_norm_s2_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=1, pad=1, sn=self.sn, scope='conv_s1_' + str(i))
                x = batch_norm(x, scope='batch_norm_s1_' + str(i))
                x = lrelu(x, 0.2)

            x = conv(x, channels=1, kernel=4, stride=1, pad=1, sn=self.sn, scope='D_logit')

            return x
    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')


        """ Input Image"""
        Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

        gpu_device = '/gpu:0'
        trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()


        self.identity_A = trainA_iterator.get_next()
        self.shape_A = trainA_iterator.get_next()
        self.other_A = trainA_iterator.get_next()

        self.shape_B = trainB_iterator.get_next()


        self.test_identity_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_identity_A')
        self.test_shape_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_shape_B')


        """ Define Generator, Discriminator """
        self.fake_same = self.generator(x_identity=self.identity_A, x_shape=self.shape_A)
        self.fake_diff = self.generator(x_identity=self.identity_A, x_shape=self.shape_B, reuse=True)
        fake_diff_shape = self.generator(x_identity=self.shape_B, x_shape=self.fake_diff, reuse=True)
        fake_diff_identity = self.generator(x_identity=self.fake_diff, x_shape=self.shape_B, reuse=True)


        real_logit = self.discriminator(x_identity=self.identity_A, x=self.other_A)
        fake_logit = self.discriminator(x_identity=self.identity_A, x=self.fake_diff, reuse=True)


        """ Define Loss """
        g_identity_loss = self.adv_weight * generator_loss(self.gan_type, fake=minpool(fake_logit)) * 64
        g_shape_loss_same = self.L1_weight * L1_loss(self.fake_same, self.shape_A)
        g_shape_loss_diff_shape = self.L1_weight * L1_loss(fake_diff_shape, self.shape_B)
        g_shape_loss_diff_identity = self.L1_weight * L1_loss(fake_diff_identity, self.fake_diff)


        self.Generator_loss = g_identity_loss + g_shape_loss_same + g_shape_loss_diff_shape + g_shape_loss_diff_identity
        self.Discriminator_loss = self.adv_weight * discriminator_loss(self.gan_type, real=real_logit, fake=fake_logit)


        """ Result Image """
        self.test_fake = self.generator(x_identity=self.test_identity_A, x_shape=self.test_shape_B, reuse=True)


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)


        """" Summary """
        self.G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)


        self.G_identity = tf.summary.scalar("G_identity", g_identity_loss)
        self.G_shape_loss_same = tf.summary.scalar("G_shape_loss_same", g_shape_loss_same)
        self.G_shape_loss_diff_shape = tf.summary.scalar("G_shape_loss_diff_shape", g_shape_loss_diff_shape)
        self.G_shape_loss_diff_identity = tf.summary.scalar("G_shape_loss_diff_identity", g_shape_loss_diff_identity)


        self.G_loss_merge = tf.summary.merge([self.G_loss, self.G_identity, self.G_shape_loss_same, self.G_shape_loss_diff_shape, self.G_shape_loss_diff_identity])
        self.D_loss_merge = tf.summary.merge([self.D_loss])


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            # linear decay
            # lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)

            if self.decay_flag :
                lr = self.init_lr * pow(0.5, epoch // self.decay_epoch)

            for idx in range(start_batch_id, self.iteration):

                train_feed_dict = {
                    self.lr : lr
                }


                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss_merge], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                real_A_images, real_B_images, fake_B_images, _, g_loss, summary_str = self.sess.run([self.identity_A, self.shape_B, self.fake_diff,
                                                                                      self.G_optim,
                                                                                      self.Generator_loss, self.G_loss_merge], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                print("Epoch: [%3d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                # display training status
                counter += 1


                if np.mod(idx+1, self.print_freq) == 0 :
                    # save_images(real_A_images, [self.batch_size, 1],
                    #             './{}/real_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    # save_images(real_B_images, [self.batch_size, 1],
                    #             './{}/real_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))
                    # save_images(fake_B_images, [self.batch_size, 1],
                    #             './{}/fake_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(np.concatenate([real_A_images, real_B_images, fake_B_images], axis=0), [self.batch_size, 3],
                                './{}/total_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx + 1))

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)



            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                          self.gan_type, self.sn,
                                          int(self.adv_weight), int(self.L1_weight))

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir) # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path) # first line
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>target</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            sample_file_name = os.path.basename(sample_file).split('.')[0]
            for target_sample_file in test_B_files :
                target_sample_image = np.asarray(load_test_data(target_sample_file))
                target_file_name = os.path.basename(target_sample_file).split('.')[0]

                image_path = os.path.join(self.result_dir,'{}_{}.png'.format(sample_file_name, target_file_name))

                fake_img = self.sess.run(self.test_fake, feed_dict = {self.test_identity_A : sample_image, self.test_shape_B : target_sample_image})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))

                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (target_sample_file if os.path.isabs(target_sample_file) else (
                        '../..' + os.path.sep + target_sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")

        index.close()