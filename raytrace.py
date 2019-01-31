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
import sys
from xml.dom import minidom
from PIL import Image, ImageDraw

xRes = 64
yRes = 48
MAX_RAY_DEPTH = 2

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
#		mapping = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
#		color += mapping[int(r/16)]
#		color += mapping[int(r%16)]
#		color += mapping[int(g/16)]
#		color += mapping[int(g%16)]
#		color += mapping[int(b/16)]
#		color += mapping[int(b%16)]
		
		color = "#{0:02x}{1:02x}{2:02x}".format(int(r*255),int(g*255),int(b*255))
		print(r,g,b)
#		color = color.replace("-","0")
		print(color)
		self.draw.point((x,y),color)

	def drawPixelColor(self,x,y,color):
		#self.drawPixelRGB(x,y,color.r,color.g,color.b)
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
class Vector:
	def __init__(self,x,y,z,w=1):
		if not (isinstance(x,float) or isinstance(x,int)):
			message = "recieved wrong type: ".format(str(x))
			raise TypeError(message)
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
#		print(type(self),type(val))
#		print(self,val)
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

	def __repr__(self):
		return "x: " + str(self.x) + " y: " + str(self.y) + " z: " + str(self.z) + " w: " + str(self.w)

class Point(Vector):
	pass

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
		self.position = center
		self.radius = radius
		self.color = color
		self.reflectivity = reflectivity
		self.transparency = transparency

	def RayCollides(self,ray,distance=1000):
		m = ray.o - self.center
		b = m.dot(ray.d)
		c = m.dot(m) - (self.radius * self.radius)

		
		#Exit if r's origin outside of the sphere (c > 0) and ray is pointing away from sphere (b > 0)
		if c > 0 and b > 0:
			return 0
		discr = b*b - c
#		print(discr,m,b,c)
		
		#a negative discriminant corresponds to ray missing sphere
		if discr < 0:
			return 0
#		print(2)
		# Ray now found to intersect sphere, compute smallest t value of intersection
		t = -b - math.sqrt(discr)

		#if t is negative, ray started inside sphere so clamp t to zero
		if (t < 0):
			t = 0
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

class Color(Vector):
	def __init__(self,r,g,b,a=1):
		self.x = r
		self.y = g
		self.z = b
		self.w = a
	
	def __repr__(self):
		return "{},{},{}".format(self.x,self.y,self.z)

class Light:
	def __init__(self,position,color):
		self.position = position
		self.color = color

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


def render(objects,lights):
	width = xRes
	height = yRes
	image = Outs("Render",xRes,yRes,"BMP",True)

	invWidth = 1.0 / width
	invHeight = 1.0 / height
	fov = 30
	aspectratio = width / float(height)
	angle = math.tan(math.pi * 0.5 * fov / 180.0);
	# Trace Rays
	for y in range(height):
		for x in range(width):
			print_status(x,y,width,height)
			xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio
			yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle
			#print("xx: {}\t yy:{}".format(xx,yy))
			ray = Ray(Point(0,0,0),Point(xx, yy, -1))
			image.drawPixelColor(x,y,trace(ray, spheres, lights, 0))
	image.save()

def print_status(x,y,width,height):
	total = width*height
	pos = 100.0 * y * height + x
	sys.stdout.write("Progress {}%\r".format(int(pos/total)))
	sys.stdout.flush()

def trace(ray,objects,lights,depth):
	min_distance = float("inf")
	collidee = None
	collision_point = None
	for obj in objects:
		result = obj.RayCollides(ray)
		if result != 0:
			distance = result[0]
			if distance < min_distance:
				collidee = obj
				collision_point = result[1]
				min_distance = distance

	if not collidee:
		return Color(0,0,0) #no collisions, return black
	
	collision_normal = collision_point - obj.position
	collision_normal.normalize()
	bias = 1e-4
	inside = False
	if ray.d.dot(collision_normal) > 0:
		collision_normal = collision_normal * (-1.0)
		inside = False

	if (collidee.transparency > 0 or collidee.reflectivity > 0) and depth < MAX_RAY_DEPTH:
		facingratio = ray.d.dot(collision_normal) * (-1.0)
		# change the mix value to tweak the effect
		fresneleffect = mix(math.pow(1 - facingratio, 3), 1, 0.1)
		# compute reflection direction (not need to normalize because all vectors
		# are already normalized)
		refldir = ray.d - collision_normal * 2 * ray.d.dot(collision_normal)
		refldir.normalize()
		reflray = Ray(collision_point + collision_normal * bias,refldir)
		reflection = trace(reflray, objects,lights, depth + 1)
		
		refraction = Point(0,0,0)
		# if the sphere is also transparent compute refraction ray (transmission)
		if collidee.transparency:
			ior = 1.1
			eta = ior if inside else 1 # are we inside or outside the surface?
			cosi = 0 - collision_normal.dot(ray.d)
			k = 1 - eta * eta * (1 - cosi * cosi)
			refrdir = ray.d * eta + collision_normal * (eta * cosi - math.sqrt(k))
			refrdir.normalize()
			refrray = Ray(collision_point + collision_normal * bias,refrdir)
			refraction = trace(refrray, objects, lights, depth + 1)

		#the result is a mix of reflection and refraction (if the sphere is transparent)
		ref = reflection * fresneleffect
		frac = refraction * (1 - fresneleffect)
		col = collidee.color * collidee.transparency
#		print("types pre addition: ", ref,frac)
		surfaceColor = ref + frac
#		print(surfaceColor,col)
		surfaceColor = surfaceColor.cross(col)
		print(surfaceColor)
		return surfaceColor
	else:
		light = lights[0]
		collision_normal = collision_point - obj.position
		lightDirection = light.position - collision_point
		lightDirection.normalize()
		rayPosition = collision_point + collision_normal * bias
		isShadow = False
		for obj in objects:
			result = obj.RayCollides(Ray(rayPosition,lightDirection))
			if result != 0:
				isShadow = True
				break
				
		if isShadow:
			return Color(0,0,0)
		else:
			return collidee.color

def mix(a,b,mix):
	return b * mix + a * (1 - mix)

def get_distance(p1,p2):
	distance = math.sqrt((p1.x-p2.x)**2+(p1.y-p2.x)**2+(p1.z-p2.z)**2)

spheres = []
spheres.append(Sphere(Point(0.0,-10004,-20),10000,Color(0.2,0.2,0.2),0,0))
spheres.append(Sphere(Point(0.0,0,-20),4,Color(1.0,0.32,0.36),1,0.5))
spheres.append(Sphere(Point(5.0,-1,-15),2,Color(0.9,0.76,0.46), 1, 0.0))
spheres.append(Sphere(Point(5.0, 0, -25), 3, Color(0.65, 0.77, 0.97), 1, 0.0))
spheres.append(Sphere(Point(-5.5,0,-15), 3, Color(0.9,0.9,0.9),1,0))
#spheres.append(Sphere(Point(0,0,-15), 3, Color(0.9,0.0,0.9),1,0))

light = Light(Point( 0.0, 20, -30), Color(0.00, 0.00, 0.00));

render(spheres,[light])
print
