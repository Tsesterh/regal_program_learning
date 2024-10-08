from program_refactoring.domains.logos.pyturtle import PyTurtle
from program_refactoring.domains.logos.pyturtle import HALF_INF, INF, EPS_DIST, EPS_ANGLE

turtle = PyTurtle()
def forward(dist):
    turtle.forward(dist)
def left(angle):
    turtle.left(angle)
def right(angle):   
    turtle.right(angle)
def teleport(x, y, theta):
    turtle.teleport(x, y, theta)
def penup():
    turtle.penup()
def pendown():
    turtle.pendown()
def position():
    return turtle.x, turtle.y
def heading():
    return turtle.heading
def isdown():
    return turtle.is_down
def embed(program, local_vars):
    # NOTE: Program must be a string, and locals() must be provided as local_vars
    # expected usage: embed("function(arg)", locals())
    return turtle.embed(program, local_vars)
from temp.codebank import *

from temp.codebank import *


turtle.save('temp/result_pred.jpg')