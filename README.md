# RIEU: Reconstruction of Stereo Images from Anaglyphs Enabled Through U-Nets

This repository follows contains code to solve the problem of recreating stereo images from 3D-Anaglyphs using Deep Learning. The goal is to create a model that can re-generate stereo images from 3D-Anaglyphs.

> The presented system is called RIEU (**R**econstructing Stereo **I**mages **E**nabled Through **U**-Nets). It is able to recreate stereo images from an input 3D-anaglyph image.
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

                ____     ____    ______   __  __
               / __ \   /  _/   / ____/  / / / /
              / /_/ /   / /    / __/    / / / / 
             / _, _/  _/ /    / /___   / /_/ /  
            /_/ |_|  /___/   /_____/   \____/   
                                    
  /// Reconstruction of Stereo Images from Anaglyphs  ///
 ///              Enabled Through U-Nets             ///
```

The following approaches have been investigated further:

## GAN (Generative Adversarial Network) Architecture

The GAN architecture is a generative model that consists of two networks: a generator and a discriminator. The generator is responsible for generating new data instances, while the discriminator is responsible for distinguishing between real and generated data instances. The generator and discriminator are trained simultaneously, with the generator trying to fool the discriminator and the discriminator trying to distinguish between real and generated data instances.

![GAN Architecture](./docs/img/gan_architecture_anaglyph.jpg)


## U-Net Architecture

The U-Net architecture is a convolutional neural network that is mainly used for image segmentation. It consists of an encoder and a decoder, with skip connections between the encoder and decoder. The encoder is responsible for extracting features from the input image, while the decoder is responsible for generating the output image. The skip connections help to preserve spatial information during the upsampling process. <br>
By transforming the image segmentation task into a pixel-wise classification task, the U-Net architecture can be used to generate inverse ana-glyphs from 3D-Anaglyphs, which can be used to recreate the final stereo images.

![U-Net Architecture](./docs/img/unet_architecture_anaglyph.jpg)

# Structure of this repository
- [data_creation/](data_creation/Readme.md): Contains scripts to create the dataset and prepare the data for training the models.
- [docker/](docker/Readme.md): Contains the Dockerfile build and run the training in a Docker container.
- [docs/](docs/Readme.md): Contains images and other files used in the README.md file, as well as a scientific one-pager of the project.
- [image_colorization/](image_colorization/README.md): Contains the code for the U-Net and GAN architecture.

# Data Sources
The trained models have been trained using the **[Holopix50k](https://leiainc.github.io/holopix50k/)** dataset:<br>
Y. Hua et al., “Holopix50k: A Large-Scale In-the-wild Stereo Image
Dataset”  in _CVPR Workshop on Computer Vision for Augmented and Virtual Reality_, Seattle, WA, 2020. Available: [http://arxiv.org/pdf/2003.11172]()

