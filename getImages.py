import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import random

img_folder = './object-detection-crowdai'
csvfile = '/labels_crowdai.csv'

temp_list=[]
car_list =[]
last_yspan = 0
last_Frame = None
empty=0

cnt = 0
n_cars = 0
new_Frame = 1
with open(img_folder + csvfile) as file:
	reader = csv.reader(file)
	for line in reader:
		cnt+=1
		if cnt ==1:
			continue
		xmin, ymin, xmax, ymax, Frame, Label = line[0:6]
		if( Frame == last_Frame):
			if Label == 'Car':		
				temp_list.append(line[0:6])
				#yspan = int(ymax) - int(ymin)
				#if yspan > last_yspan: # check if yspan_new = yspan old
				#	temp_list = line[0:6]					
				#	last_yspan = yspan
		else:
			if cnt > 1:
				list_size = len(temp_list)
				if list_size == 0:
					empty +=1
					last_Frame = Frame
					temp_list = []
				else:
					index  = random.randint(0,list_size-1)
					car_list.append(temp_list[index])
					last_Frame = Frame
					temp_list = []

				
print('emtpy lists: ', empty)
print('Total Images: ',len(car_list))
print('Total Lines: ', cnt)


# read image segments from images, resize to 64 x 64 and save as .png
output_dir = 'vehicles_UD/'

#print('\n'*2, car_list[0:3])

print('\n'*2)

cnt = 0
for line in car_list:
	if (line != []):	
		xmin, ymin, xmax, ymax, Frame, Label = line
		#print(xmin, ymin, xmax, ymax, Frame, Label); print('\n')

		img = cv2.imread(img_folder + '/' + Frame)
		
		#if(img != []):
		img_seg = img[int(ymin):int(ymax), int(xmin):int(xmax)]
		img_seg = cv2.resize(img_seg, (64,64))
		out_name = output_dir + Frame
		print(out_name)
		cv2.imwrite(out_name, img_seg)
	cnt +=1


print('Done!')
"""
img = cv2.imread(img_folder + '/'+ Frame)
print('read ret: ',img)
plt.figure()
plt.imshow(img[int(ymin):int(ymax),int(xmin):int(xmax)])
plt.show()
""" 
