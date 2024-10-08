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

for j in range(7):
    embed("""penup()
forward(8)
left(0.0)
pendown()
for i in range(4):
    forward(2)
    left(90.0)""", locals())
    forward(0)
    left(51.42857142857143)

turtle.save('temp/result_gold.jpg')