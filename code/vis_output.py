import cv2

img_name = 'C:/Users/5as4as/Desktop/007-1789-100'
label = cv2.imread(img_name + '.png', flags=0)
img = cv2.imread(img_name + '.jpg')


disk_label = label
disk_label[disk_label==128] = 255
contours, _  = cv2.findContours(disk_label, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, 0, (0,0,255), 2)
cv2.imwrite(img, )