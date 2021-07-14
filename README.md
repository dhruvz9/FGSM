# TENSORFLOW IMPLEMENTATION OF FGSM

## usage
```
python fgsm.py
```
## Help Log
```
Usage: ipykernel_launcher.py [-h]
optional arguments:
  -h, --help            show this help message and exit
  
```                        
## Contributers:
- [Pranjal Sharma](https://github.com/sppsps)
- [Dhrubajit Basumatary](https://github.com/dhruvz9)

## REFERENCE
 - Title : Explaining and Harnessing Adversarial Examples <br />
 - Link : https://arxiv.org/abs/1412.6572 <br />
 - Author : Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy <br />
 - Published : 20 Mar 2015  <br />
 
 # Summary
 The fast gradient sign method works by using the gradients of the neural network to create an adversarial example. For an input image, the method uses the gradients of the loss with respect to the input image to create a new image that maximises the loss. This new image is called the adversarial image. This can be summarised using the following expression:
 ![alt_text](https://www.pyimagesearch.com/wp-content/uploads/2021/02/fgsm_equation.png)<br>
 where:
 <ul>
  <li> {adv}\_x: Our output adversarial image</li>
  <li>  x: The original input image</li>
  <li> y: The ground-truth label of the input image</li>
   <li> \epsilon: Small value we multiply the signed gradients by to ensure the perturbations are small enough that the human eye cannot detect them but large enough that they fool the neural network</li>
  <li> \theta: Our neural network model</li>
  <li> J: The loss function</li>
 </ul>
 # Example
 ![alt text](https://www.researchgate.net/publication/336402462/figure/fig1/AS:812471887609871@1570719801771/An-adversarial-example-generated-by-the-FGSM-attack-16-on-the-VGG-16-network-55.jpg)
