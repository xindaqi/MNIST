from turtle import *


def flower():
	color("red", "red")
	begin_fill()
	for _ in range(50):
		forward(200)
		left(170)
	end_fill()

	mainloop()


def triangle():
	for i in range(3):
		forward(200)
		right(120)

	mainloop()


class Doraemon(object):
	def my_goto(self, x, y):
		penup()
		goto(x, y)
		pendown()
	def eyes(self):
		tracer(False)
		a = 2.5
		for i in range(120):
			if 0 <= i < 30 or 60 <= i <90:
				a -= 0.05
				lt(3)
				fd(a)
			else:
				a += 0.05
				lt(3)
				fd(a)
		tracer(True)
	def beard(self):
		self.my_goto(-37, 135)
		seth(165)
		fd(60)

		self.my_goto(-37, 125)
		seth(180)
		fd(60)

		self.my_goto(-37, 115)
		seth(193)
		fd(60)

		self.my_goto(37, 135)
		seth(15)
		fd(60)

		self.my_goto(37, 125)
		seth(0)
		fd(60)

		self.my_goto(37, 115)
		seth(-13)
		fd(60)

	def mouth(self):
		self.my_goto(5, 148)
		seth(270)
		fd(100)
		seth(0)
		circle(120, 50)
		seth(230)
		circle(-120, 100)

	def scarf(self):
		
		fillcolor('#e70010')
		begin_fill()
		seth(0)
		fd(200)
		circle(-5, 90)
		fd(10)
		circle(-5, 90)
		fd(207)
		circle(-5, 90)
		fd(10)
		circle(-5, 90)
		end_fill()

	def nose(self):
		self.my_goto(-10, 158)
		fillcolor('#e70010')
		begin_fill()
		circle(20)
		end_fill()

	def black_eyes(self):
		seth(0)
		self.my_goto(-20, 195)
		fillcolor('#000000')
		begin_fill()
		circle(13)
		end_fill()

		pensize(6)
		self.my_goto(20, 205)
		seth(75)
		circle(-10, 150)
		pensize(3)

		self.my_goto(-17, 200)
		seth(0)
		fillcolor('#ffffff')
		begin_fill()
		circle(5)
		end_fill()
		self.my_goto(0, 0)

	def face(self):
		fd(183)
		fillcolor('#ffffff')
		begin_fill()
		lt(45)
		circle(120, 100)

		seth(90)
		self.eyes()
		seth(180)
		penup()
		fd(60)
		pendown()
		seth(90)
		self.eyes()
		penup()
		seth(180)
		fd(64)
		pendown()
		seth(215)
		circle(120, 100)
		end_fill()
	def head(self):
		penup()
		circle(150, 40)
		pendown()
		fillcolor('#00a0de')
		begin_fill()
		circle(150, 280)
		end_fill()
	def haha(self):
		self.head()
		self.scarf()
		self.face()
		self.nose()
		self.mouth()
		self.beard()
		self.my_goto(0, 0)
		seth(0)
		penup()
		circle(150, 50)
		pendown()
		seth(30)
		fd(40)
		seth(70)
		circle(-30, 270)

		fillcolor('#00a0de')
		begin_fill()

		seth(230)
		fd(80)
		seth(90)
		circle(1000, 1)
		seth(-89)
		circle(-1000, 10)

		seth(180)
		fd(70)
		seth(90)
		circle(30, 180)
		seth(180)
		fd(70)

		seth(100)
		circle(-1000, 9)

		seth(-86)
		circle(1000, 2)
		seth(230)
		fd(40)

		circle(-30, 230)
		seth(45)
		fd(81)
		seth(0)
		fd(203)
		circle(5, 90)
		fd(10)
		circle(5, 90)
		fd(7)
		seth(40)
		circle(150, 10)
		seth(30)
		fd(40)
		end_fill()

		# left hand
		seth(70)
		fillcolor('#ffffff')
		begin_fill()
		circle(-30)
		end_fill()

		# foot
		self.my_goto(103.74, -182.59)
		seth(0)
		fillcolor('#ffffff')
		begin_fill()
		fd(15)
		circle(-15, 180)
		fd(90)
		circle(-15, 180)
		fd(10)
		end_fill()

		self.my_goto(-96.26, -182.59)
		seth(180)
		fillcolor('#ffffff')
		begin_fill()
		fd(15)
		circle(15, 180)
		fd(90)
		circle(15, 180)
		fd(10)
		end_fill()

		# right hand
		self.my_goto(-133.97, -91.81)
		seth(50)
		fillcolor('#ffffff')
		begin_fill()
		circle(30)
		end_fill()

		# bag
		self.my_goto(-103.42, 15.09)
		seth(0)
		fd(38)
		seth(230)
		begin_fill()
		circle(90, 260)
		end_fill()

		self.my_goto(5, -40)
		seth(0)
		fd(70)
		seth(-90)
		circle(-70, 180)
		seth(0)
		fd(70)

		# ding
		self.my_goto(-103.42, 15.09)
		fd(90)
		seth(70)
		fillcolor('#ffd200')
		begin_fill()
		circle(-20)
		end_fill()
		seth(170)
		fillcolor('#ffd200')
		begin_fill()
		circle(-2, 180)
		seth(10)
		circle(-100, 22)
		circle(-2, 180)
		seth(170)
		circle(100, 22)
		end_fill()

		goto(-13.42, 15.09)
		seth(250)
		circle(20, 110)
		seth(90)
		fd(15)
		dot(10)
		self.my_goto(0, -150)
		self.black_eyes()





if __name__ == "__main__":
	screensize(800, 600, '#f0f0f0')
	pensize(3)
	speed(9)

	d_e = Doraemon()
	d_e.haha()
	d_e.my_goto(100, -300)
	mainloop()

	# triangle()
	# flower()

