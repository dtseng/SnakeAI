# SnakeAI
This is an agent trained to play Snake using my implementation of NEAT (Neuroevolution of Augmenting Topologies), an algorithm described in Dr. Stanley's paper [(here)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). NEAT is a type of neuroevolution that aims to evolve both the weights and structure of the player's neural network over the course of many generations. The player gradually develops its own strategies to increase its fitness.

Here is an example progression of the player's strategy:



**Generation 0:**

![](https://zippy.gfycat.com/FondDangerousGonolek.gif)

**Generation 100:**

![](https://zippy.gfycat.com/PhonyAgedAmericanavocet.gif)

**Generation 200:**

![](https://zippy.gfycat.com/AnchoredRectangularFerret.gif)


(With more input information and larger population size, the player can develop more complex strategies. However, the training time will increase significantly.) 

# To run this program:
```
python evolve_snake.py
```
