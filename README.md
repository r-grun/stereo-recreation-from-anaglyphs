# ReStereo: Recreation of stereo Images from 3D-Anaglyphs using Deep Learning

This repository follows contains code to solve the problem of recreating stereo images from 3D-Anaglyphs using Deep Learning. The goal is to create a model that can re-generate stereo images from 3D-Anaglyphs.

> The presented system is called ReStereo. It is able to recreate stereo images from an input 3D-anaglyph image.
``` 
                                      _.....__
                             (.--...._`'--._
                   _,...----''''`-.._ `-..__`.._
          __.--'-;..-------'''''`._')      `--.-.__
        '-------------------------------------------'
        \ '----------------  ,-.  .-------------'. |
         \`.              ,','  \ \             ,' /
          \ \             / /   `.`.          ,' ,'
          `. `.__________/,'     `.' .......-' ,'
            `............-'        "---------''

            ______     _____ _
            | ___ \   /  ___| |
            | |_/ /___\ `--.| |_ ___ _ __ ___  ___
            |    // _ \`--. \ __/ _ \ '__/ _ \/ _ \
            | |\ \  __/\__/ / ||  __/ | |  __/ (_) |
            \_| \_\___\____/ \__\___|_|  \___|\___/

        --- Recreate Stereo Images from its Anaglyph! ---
```

The following approaches have been investigated further:

## GAN (Generative Adversarial Network) Architecture

The GAN architecture is a generative model that consists of two networks: a generator and a discriminator. The generator is responsible for generating new data instances, while the discriminator is responsible for distinguishing between real and generated data instances. The generator and discriminator are trained simultaneously, with the generator trying to fool the discriminator and the discriminator trying to distinguish between real and generated data instances.

![GAN Architecture](./docs/img/gan_architecture_anaglyph.jpg)


## U-Net Architecture

The U-Net architecture is a convolutional neural network that is mainly used for image segmentation. It consists of an encoder and a decoder, with skip connections between the encoder and decoder. The encoder is responsible for extracting features from the input image, while the decoder is responsible for generating the output image. The skip connections help to preserve spatial information during the upsampling process. <br>
By transforming the image segmentation task into a pixel-wise classification task, the U-Net architecture can be used to generate inverse ana-glyphs from 3D-Anaglyphs, which can be used to recreate the final stereo images.

![U-Net Architecture](./docs/img/unet_architecture_anaglyph.jpg)