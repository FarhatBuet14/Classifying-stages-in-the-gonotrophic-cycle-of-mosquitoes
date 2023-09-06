## Classifying Stages in the Gonotrophic Cycle of Mosquitoes
We have designed computer vision techniques to determine stages in the gonotrophic cycle (unfed, fully fed, semi-gravid and gravid)) of female mosquitoes (*Aedes aegypti, Anopheles stephensi, and Culex quinquefasciatus*) from images captured by smartphones.

See the paper [here](https://assets.researchsquare.com/files/rs-3191730/v1_covered_6cfac5b8-31ac-4e10-897c-b692ac1255ff.pdf?c=1690863911)

## Abdominal Conditions of a Female Mosquito According to the Stages of its Gonotrophic Cycle

![gonotrphic_cycle.png](https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/blob/main/images/gonotrphic_cycle.png)

## Dataset Details

![dataset.png](https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/blob/main/images/dataset.png)

## Data Augmentation Techniques

![augmentation.png](https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/blob/main/images/augmentation.png)

## Requirements
* Python 3.10.12
* Tensorflow 2.12
* Keras 2.12

## Folder Details

| Folder       | Description                                                               |
|--------------|---------------------------------------------------------------------------|
| `codes/`     | Provides the source code.                                                 |
| `data/`      | Contains the dataset - training (with augmentation), validation, test.    |
| `models/`    | Saves training models according to the model architecture.                |


## Installation
~~~~{.python}
git clone https://github.com/FarhatBuet14/Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes.git
cd Classifying-stages-in-the-gonotrophic-cycle-of-mosquitoes/codes
pip install -r requirements.txt
~~~~

## How To Run

#### * Training
For Training run the *train.py* and provide a model name i.e. *EfficientNetB0*. 
~~~~{.python}
python train.py --name EfficientNetB0
~~~~
Other parameters can be passed as arguments. 
~~~~{.python}
python train.py --name EfficientNetB0 --ep 500 --batch 16 
~~~~


#### * Test an Image and Generate Grad-CAM

To test a trained model with an image, place the test image in the current folder, rename the image to *test_image.jpg*, put the model directory link to the *model* variable in *test_image.py* file and then run *test_image.py* file. 
~~~~{.python}
python test_image.py
~~~~

It will print the prediction with the confidence and generate the Grad-CAM which will be saved to the current folder with the name *gradCam_test_image.jpg*

#### * Generate the TSNE Plot

To generate a TSNE plot with a trained model, put the model directory link to the *model* variable in *tsne.py* file and then run *tsne.py* file. 

~~~~{.python}
python tsne.py
~~~~

It will generate the TSNE plot which will be saved to the current folder with the name *tsne.png*
