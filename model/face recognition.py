#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import face_recognition
import os
import glob
import time


# In[2]:


#pip install matplotlib


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[4]:


img = mpimg.imread(r"H:\face recognition\photos\amr zaki.jpg")
imgplot = plt.imshow(img)
plt.show()


# In[5]:


start = time.time()

# Specify the directory path where your photos are located
directory_path = r"H:\face recognition\photos/"

# Use glob to retrieve a list of file paths for all photos in the directory
photo_paths = glob.glob(directory_path + "*.jpg")  # Modify the file extension if needed

# Loop through the photo paths
for photo_path in photo_paths:
    # Process each photo as needed
    # For example, you can load the photo using OpenCV and perform face recognition
    known_image = face_recognition.load_image_file(r"H:\face recognition\test\test23.jpg")
    unknown_image = face_recognition.load_image_file(photo_path)

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    if(results==[True]):
        photo_path = os.path.split(photo_path)
        print(photo_path[1].split('.jpg')[0])

img = mpimg.imread(r"H:\face recognition\test\test23.jpg")
imgplot = plt.imshow(img)
plt.show()
end = time.time()
print(end - start)

