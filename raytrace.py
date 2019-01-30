###################################################################################################
# This is a simple ray trace program. The point of it is to gain a better understanding of how    #
# ray-tracing works. I'm programming it first in python, and then I'll move later to c with simd  #
# to speed things up. I'm using this as an exercise in 3D physical systems. I want to gain an     #
# understanding of systems like colision dtection, bounding boxes, octrees, etc, in a system that #
# is much less complex first before I move on to implementing them as part of a physics           #
# simulation that will need to be running in openCL on the GPU.                                   #
# I plan on doing this in 3 phases. phase 1 is simple and straight forward. ray casts out till it #
# hits geometry, calculate from there to the lights to get the appropriate coloring. Phase 2 I    #
# will fix it up for reflecting n times, where n is a value specified when the job is dispatched. #
# Phase 3 i flip the system on its head. each light sends out hundreds of rays and paint the      #
# world, then rays our sent from where ever the collide to the camera and the pixels are added    #
# till a picture is created. 									  #
###################################################################################################

import math
import random
import time
from xml.dom import minidom
from PIL import Image, ImageDraw

xRes = 64
yRes = 48

PI180 = 0.0174532925

gcams = []
gmesh = []
gmats = []

class Outs:
	def __init__(self,name,sizeX,sizeY,ext,counting):
		self.img = Image.new("RGB",(sizeX,sizeY))
		self.draw = ImageDraw.Draw(self.img)
		self.name = name
		self.ext = ext
		if counting:
			self.count = 0
		else:
			self.count = -1

	def drawPixelRGB(self,x,y,r,g,b):
		color = "#"
		mapping = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
		color += mapping[r/16]
		color += mapping[r%16]
		color += mapping[g/16]
		color += mapping[g%16]
		color += mapping[b/16]
		color += mapping[b%16]
		self.draw.point((x,y),color)

	def drawPixelColor(self,x,y,color):
		self.drawPixelRGB(x,y,color.x,color.y,color.z)

	def drawPixelHash(self,x,y,val):
		self.draw.point((x,y),val)

	def save(self):
		if self.count != -1:
			self.img.save(self.name + str(self.count) + "." + self.ext,self.ext)
			self.count += 1
		else:
			self.img.save(self.name + "." + self.ext,self.ext)

#####	Geometric Objects	###################################################################
class Point:
	def __init__(self,x,y,z,w=1):
		self.x = x * 1.0
		self.y = y * 1.0
		self.z = z * 1.0
		self.w = w * 1.0

	def __add__(self,val):
		x = val.x + self.x
		y = val.y + self.y
		z = val.z + self.z
		return Point(x,y,z)

	def __neg__(self):
		return Point(-self.x,-self.y,-self.z)
		
	def __sub__(self,val):
		x = self.x - val.x
		y = self.y - val.y
		z = self.z - val.z
		w = self.w - val.w
		return Point(x,y,z,w)

	def __mul__(self,val):
		x = val * self.x
		y = val * self.y
		z = val * self.z
		w = self.w
		return Point(x,y,z,w)
	
	def mag2(self):
		return self.x**2 + self.y**2 + self.z**2		

	def mag(self):
		return math.sqrt(self.mag2())
		
	def dot(self,val):#dot product
		x = self.x * val.x
		y = self.y * val.y
		z = self.z * val.z
		w = self.w * val.w
		return x + y + z

	def cross(self,val):#cross product
		x = self.y * val.z - self.z * val.y
		y = -(self.x * val.z - self.z * val.x)
		z = self.x * val.y - self.y * val.x
		p = Point(x,y,z)
#		p.normalize()
		return p

	def normalize(self):
		self.w = self.mag()
		if self.w == 0:
			self.w = 1
			self.x /= self.w
			self.y /= self.w
			self.z /= self.w
			self.w = 0
		else:
			self.x /= self.w
			self.y /= self.w
			self.z /= self.w
			self.w = 1

	def __str__(self):
		return "x: " + str(self.x) + " y: " + str(self.y) + " z: " + str(self.z) + " w: " + str(self.w)

class Triangle:
	def __init__(self,A,B,C):
		self.A = A
		self.B = B
		self.C = C
		AB = B - A
		AC = C - A
		self.n = AB.cross(AC)
		self.n.normalize()
		self.d = self.n.dot(A)

	def LineCollidesPlane(self,val):
		AB = val.B - val.A
		t = (self.d - (self.n.dot(val.A))) / (self.n.dot(AB))

		# if t in [0...1] compute and return intersectin point
		if (t >= 0 and t <= 1):
			return val.A + (AB * t)
		return None

	def RayCollides(self,val,distance=1000):
		foo = val.lineSeg(distance)
		return IntersectLineTriangle(foo.A,foo.B,self.A,self.B,self.C)

	def __contains__(self,val):
		return self.RayCollides(val)

class Sphere:
	def __init__(self,center,radius,color,reflectivity,transparency):
		self.center = center
		self.radius = radius
		self.color = color
		self.reflectivity = reflectivity
		self.transparency = transparency

	def RayCollides(self,ray,distance=1000):
		m = ray.o - self.center
		b = m.dot(ray.d)
		c = m.dot(m) - (self.radius * self.radius)
#		print(m,b,c)
		
		#Exit if r's origin outside of the sphere (c > 0) and ray is pointing away from sphere (b > 0)
		if c > 0 and b > 0:
			return 0
#		print(1)
		discr = b*b - c
		
		#a negative discriminant corresponds to ray missing sphere
		if discr < 0:
			return 0
#		print(2)
		# Ray now found to intersect sphere, compute smallest t value of intersection
		t = -b - math.sqrt(discr)

		#if t is negative, ray started inside sphere so clamp t to zero
		if (t < 0):
			t = 0
#		print(3)
		q = ray.o + ray.d * t
		return (t,q)

	def __contains__(self,val):
		return self.RayCollides(val)

class LineSegment:
	def __init__(self,A,B):
		self.A = A
		self.B = B

class Ray:
	def __init__(self,Origin,Direction):
		self.o = Origin
		self.d = Direction
		self.d.normalize()

	def lineSeg(self,length):
		return LineSegment(self.o,self.o + self.d*(length*1.0))

class Intersection:
	def __init__(self,depth,point,plane):
		self.depth = depth
		self.point = point
		self.plane = plane

class Camera:
	def __init__(self,loc,direction,clipNear,clipFar,FOV,sizeX,sizeY,name):
		self.loc = loc
		self.d = direction
		self.cn = clipNear
		self.cf = clipFar
		self.fov = FOV
		self.x = sizeX
		self.y = sizeY
		self.aspect = sizeY/(sizeX*1.0)
		self.name = name
		self.outs = Outs(name,sizeX,sizeY,"BMP",True)
		self.xspan = math.tan((FOV/2*PI180))*clipNear*2
		self.yspan = self.xspan*self.aspect
		self.xpitch = self.xspan/sizeX
		self.ypitch = self.yspan/sizeY

	def draw(self,world):
		tstart = time.time()
		xguide = Point(0,1,0).cross(self.d)
		yguide = self.d.cross(xguide)
		xguide.normalize()
		yguide.normalize()
		#print "xguide:", str(xguide)
		#print "yguide:", str(yguide)
		rayplane = self.loc + self.d * self.cn
		start = rayplane - xguide * (self.xspan/2)
		start = start - yguide * (self.yspan/2)
		pstart = time.time()
		
		for x in range(self.x):
			tempx = (self.xpitch * x) + self.xpitch/2
			print (x*100.0)/self.x,"percent complete, row took:",time.time()-pstart
			pstart = time.time()
			for y in range(self.y):
				tempy = (self.ypitch * y) + self.ypitch/2
				castpoint = start + xguide * tempx + yguide * tempy
#				print "cast point is:", str(castpoint)
#				print castpoint.x,',',castpoint.y
				castdir = castpoint - self.loc
				castdir.normalize()
				castRay = Ray(self.loc,castdir)
				intersects = []
				for tri in world:
					col = tri.RayCollides(castRay,self.cf)
					if col:
						intersects.append(col)
				if len(intersects) == 0:
					self.outs.drawPixelRGB(x,y,0,0,0)
				else:
					self.outs.drawPixelRGB(x,y,255,255,255)
		self.outs.save()
		print "job took:", time.time()-tstart

class Color:
	def __init__(self,r,g,b,a=1):
		self.r = r
		self.g = g
		self.b = b
		self.a = a

class Mesh:
	def __init__(self):
		self.verts = []
		self.faces = []

	def addVert(self,vert):
		self.verts.append(vert)

	def addFace(self,a,b,c):
		self.faces.append(Triangle(self.verts[a],self.verts[b],self.verts[c]))

class Light:
	def __init__(self,position,color):
		self.position = position
		self.color = color

def drawRandom(x, y, outs):
	outs.drawPixelRGB(x,y,random.randint(0,15)*16,random.randint(0,15)*16,random.randint(0,15)*16)

def drawWorld(outs,function):
	for i in range(outs.img.size[0]):
		for j in range(outs.img.size[1]):
			function(i,j,outs)

def scalarTriple(a,b,c):
	return a.cross(b).dot(c)

def IntersectLineTriangle(x,y,a,b,c):
	xy = y - x
	pl = Triangle(a,b,c)

	if pl.n.dot(xy):
		t = (pl.d - pl.n.dot(x))/ pl.n.dot(xy)
	else:
		t = (pl.d - pl.n.dot(x))

	if not(t >= 0 and t <= 1): return None

	p = x + (xy * t)
#	print t,"with",str(p)
	a -= p
	b -= p
	c -= p

	ab = a.dot(b)
	ac = a.dot(c)
	bc = b.dot(c)
	cc = c.dot(c)
	if (bc * ac - cc * ab < 0): return None
	bb = b.dot(b)
	if (ab * bc - ac * bb < 0): return None
	return p


#A = Point( 00, 00,-1000.0)
#B = Point( 00, 00, 1000.0)
#R = LineSegment(A,B)
#P1= Point( 0,  1, 0)
#P2= Point( 1, -1, 0)
#P3= Point(-1, -1, 0)
#P = Triangle(P1,P2,P3)
##s = Sphere(Point(0,0,0),1)
#p = Point(10,0,0)
#d = Point(-1,0,0)

spheres = []
spheres.append(Sphere(Point(0.0,-10004,-20),10000,Color(0.2,0.2,0.2),0,0))
spheres.append(Sphere(Point(0.0,0,-20)4,Color(1.0,0.32,0.36),1,0.5))
spheres.append(Sphere(Point(5.0,-1,-15),2,Color(0.9,0.76,0.46), 1, 0.0))
spheres.append(Sphere(Point(5.0, 0, -25), 3, Color(0.65, 0.77, 0.97), 1, 0.0))
spheres.append(Sphere(Point(-5.5,0,-15), 3, Color(0.9,0.9,0.9),1,0))

light = Light(Point( 0.0, 20, -30), Color(0.00, 0.00, 0.00));
c = Camera(Point(0,0,2),Point(0,0,-1),.1,10000,90,xRes,yRes,"dotanuki out")
#c.draw(loadObj('','level.obj'))
c.draw([s])
