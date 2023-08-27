# FUCKakte
Introducing 'Fuckakte'! an engaging and lively party game that pits two teams against each other in a creative challenge. <br/> 
Each team must skillfully contort their bodies to replicate a shape presented on the screen. This exciting and entertaining game brings out players' artistic sides while encouraging friendly competition. <br/> 

![Alt Text](https://github.com/itsikshteinberger/FUCKakte/blob/main/sim/simolation.jpg)
> Me, a failed player, trying to make the shape of the Eiffel Tower.

# The game
Utilizing the game is remarkably straightforward. With each turn, a shape and a progress bar will emerge on the screen. As time winds down, the team's task is to embody the displayed shape using their bodies. Once the shape is recreated, the turn transitions to the opposing team, who faces the same challenge. At the conclusion of each round, the computer discerns the victorious team.

![Alt Text](https://github.com/itsikshteinberger/FUCKakte/blob/main/sim/end.png)
> You, at the end of the game, illustration

# The algorithm

* An image of an object on a white background becomes a binary image using the following steps
1. The loaded image is converted from its original color format to grayscale. Grayscale images contain only intensity values, which simplifies further processing.
2. Thresholding is applied to the grayscale image using a predefined threshold value of 220. This operation segments the image into two categories: foreground and background. Pixels with intensity values above the threshold become white (foreground), and those below become black (background).
3. A structuring element is created, which acts as a reference for morphological operations. In this case, a rectangular 3x3 structuring element is used.
4. An erosion operation is performed on the binary image. Erosion reduces small noise and fine details while maintaining the main structure of shapes.
5. Following the erosion, a dilation operation is carried out. Dilation enlarges the shapes in the image while maintaining their core features.

![Alt Text](https://github.com/itsikshteinberger/FUCKakte/blob/main/sim/Figure_1.png)

* The players are also transformed into binary images using a pretrained segmentation model called YOLO-7

![Alt Text](https://github.com/itsikshteinberger/FUCKakte/blob/main/sim/yolo.png)

* Finally, the computer calculates the players' performance using cross correlation between the two images.

$$
\text{Cross-Correlation}(x, y) = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} A(i, j) \cdot B(i - x, j - y)
$$

> The cross correlation equation

<br/> <br/>
* The game is written in pygame, before using make sure you have all the necessary libraries. <br/>
* You can also add images of objects on a white background of your choice. <br/>
* The game won the 2023 BMDC Hackathon! <br/>
Enjoy :)




