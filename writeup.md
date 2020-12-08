# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[solidWhiteCurve]: ./test_images/solidWhiteCurve.jpg
[solidWhiteCurveResult]: ./test_images_output/solidWhiteCurve.jpg

---

### Reflection

### 1. Pipeline Description.

First, I converted the images to grayscale. Converting to grayscale reduces the number
of dimensions functions in the steps to come have to work against. 

```python
# convert the images colors to grayscale
image = grayscale(image)
```

As another set up step, I apply a gaussian blur to smooth out any rogue coloring. This
Will help with canny edge detection.

```python
# run gaussian blurr to smoothe the image for gradient calculations
KERNAL_SIZE = 7
image = gaussian_blur(image, KERNAL_SIZE)
```

The next step is edge detection, using the canny edge detection algorithm. Our work in the grayscale transformation
pays off here, as this step will work to calculate color difference (which is now 1 dimensional) between pixels and
keep only those with a gradient within our provided threshold. 

```python
# use canny edge detection to detect gradient changes
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150
image = canny(image, LOW_THRESHOLD, HIGH_THRESHOLD)
```

I then remove any pixels outside our area of interest, so the noise doesn't disturb canny edge detection. 

```python
# crop out mask for image
ROI_Polygon = [[(0,height), (width/2, 5.8*height/10), (width, height), (0, height)]]
ROI = np.array(ROI_Polygon, dtype=np.int32)
image = region_of_interest(image, ROI)
```


I then modified the draw lines function to split lines into ones that have a positive slope and a negative.
Lines with positive slopes relate to the left lane line and lines with negative slopes relate to the right lane line. 

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    pos_slope_x = []
    pos_slope_y = []
    neg_slope_x = []
    neg_slope_y = []
    width = img.shape[1]
    height = img.shape[0]
    min_y = (height*.61)
    new_lines = []
    
    # bucket the lines based on pos or neg slope
    for line in lines:
        for x1,y1,x2,y2 in line:
            # find all lines that have positive or negative slopes
            slope = (y2-y1)/(x2-x1)
            if slope > 0:
                pos_slope_x.append(x1)
                pos_slope_x.append(x2)
                pos_slope_y.append(y1)
                pos_slope_y.append(y2)
            else:
                neg_slope_x.append(x1)
                neg_slope_x.append(x2)
                neg_slope_y.append(y1)
                neg_slope_y.append(y2)
```

I added the following section just in case edge detection didn't work for a frame.

```python
    # error handling just return if no lines were detected in the frame
    if len(pos_slope_x) == 0 or len(pos_slope_y) == 0 or len(neg_slope_x) == 0 or len(neg_slope_y) == 0:
        return img
```

I then did a linear fit on both left and right lane lines arrays to get the m and b values for the y = mx + b linear fit.

```python
    # get the slope and y intercept of lines     
    pos_m, pos_b = np.polyfit(pos_slope_x, pos_slope_y, 1)
    neg_m, neg_b = np.polyfit(neg_slope_x, neg_slope_y, 1)
```

I then wanted to draw a line from the bottom of the image to the top of my area of interest. So I used the linear fit 
function with my new m and b values along with either the height or min_y variable as the y to calculate my x values for 
the provided intercept. This gave me the points I needed to plot the detected lane line.

```python
    # get the upper bound intercept as top point
    pos_x1 = int((min_y - pos_b)/pos_m)
    pos_y1 = int(min_y)
    neg_x1 = int((min_y - neg_b)/neg_m)
    neg_y1 = int(min_y)
    
    # get the lower bound intercepts
    # find the x intercept, if the intercept is out of bounds it must hit the right bound of the image
    #   so we find that right bounding line intercept
    neg_y2 = int(height)
    neg_x2 = int((neg_y2 - neg_b)/neg_m)
    if neg_x2 >= width:
        neg_x2 = int(width - 1)
        neg_y2 = int((neg_m * neg_x2) + neg_b)
    # same as the previous section, looking for intercepts
    pos_y2 = int(height)
    pos_x2 = int((pos_y2 - pos_b)/pos_m)
    if pos_x2 < 0:
        pos_x2 = 1
        pos_y2 = int((pos_m * pos_x2) + pos_b)
```

I then used these four generated points to draw two lines.

```python
    # draw the two lines for the lane
    cv2.line(img, (pos_x1, pos_y1), (pos_x2, pos_y2), color, thickness)
    cv2.line(img, (neg_x1, neg_y1), (neg_x2, neg_y2), color, thickness)
```

Now that the drawlines works as expected, I can call hough transform which will detect lines in the image and pass 
that list of lines to the drawlines function.

```python
# run hough transform to detect lines
rho = 2
theta = np.pi/180
threshold = 34
min_line_length = 14
max_line_gap = 3
image = hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap)
```

So now I have my image, with detected lines drawn as two lines on my blank image. Now I overlay this blank image with
lane lines on top of the original image to complete the pipeline.

```python
# save the image with the lines over the original
image = weighted_img(image, orig_image)
```

Running this image through the pipeline ...

![Input][solidWhiteCurve]

Results in ...

![Output][solidWhiteCurveResult]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be driving down a windy road. The hough transform would not perform as well if the
road is not a straight line. Issues could also come about if lines on the road are not painted consistently or at all.

### 3. Suggest possible improvements to your pipeline


A possible improvement would be to average out lines between frames in a video to make the edge detection smoother.
The lines on the video wobble a lot and, it'd be much smoother if I averaged the m and b values between frames. A
change like this should not reduce the performance of the edge detection because the frame rate of the video is much higher
than the positional change of the car. 
