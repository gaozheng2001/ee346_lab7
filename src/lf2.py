#!/usr/bin/env python

from turtle import distance, right

from cv2 import sqrt
import rospy, cv2, cv_bridge, numpy, math, time, pyttsx
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber('camera/image/compressed',
                CompressedImage, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                Twist, queue_size=10)
        self.twist = Twist()
        self.stopped=False
        self.init=0
        self.start=0
        self.right=0
        self.starttimer=rospy.get_time()
        # self.lasttime
        self.finish=0
        self.time = rospy.get_time()

    def turn_detect(self , image):
        detector = 0
        h, w = image.shape
        image = image[h/2:h , 0:w]
        #cv2.imshow('test' , image)
        M = cv2.moments(image)
        if M['m00']<=3:
            detector = 1
            #print(M['m00'])
        return detector


    def turn(self):
        self.twist.linear.x = 0
        self.twist.angular.z = 0
        self.cmd_vel_pub.publish(self.twist)
        time.sleep(0.2)
        self.twist.linear.x = 0.2
        self.twist.angular.z = 2.5
        self.cmd_vel_pub.publish(self.twist)
        time.sleep(1)

    def image_callback(self, msg):
        desired_aruco_dictionary = "DICT_6X6_1000"
        # The different ArUco dictionaries built into the OpenCV library.
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }
        lower_black=numpy.array([0,0,0])
        upper_black=numpy.array([180,255,55])
        cameraMatrix = numpy.matrix(numpy.array([[258.69695, 0.0, 162.59072],
                                                 [0.0, 259.0159, 137.3223],
                                                 [0.0, 0.0, 1.0]]))
        distCoeffs = numpy.array([0.201841, -0.192094, 0.028005, 0.015335, 0.000000])
        this_aruco_dictionary = cv2.aruco.Dictionary_get(
                                    ARUCO_DICT[desired_aruco_dictionary])
        this_aruco_parameters = cv2.aruco.DetectorParameters_create()

        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            image, this_aruco_dictionary, parameters=this_aruco_parameters)

        if len(corners) > 0:
            # Flatten the ArUco IDs list
            ids = ids.flatten()
            for (marker_corner, marker_id) in zip(corners, ids):
                # Extract the marker corners
                r, t, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corner, 0.05, cameraMatrix, distCoeffs)

                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                # Convert the (x,y) coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # Draw the bounding box of the ArUco detection
                cv2.line(image, top_left, top_right, (0, 255, 0), 2)
                cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)

                # Calculate and draw the center of the ArUco marker
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw the ArUco marker ID on the video frame
                # The ID is always located at the top_left of the ArUco marker
                cv2.putText(image, str(marker_id),
                            (top_left[0], top_left[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

                #import pdb;pdb.set_trace()
                dis_z = numpy.abs((t[0][0][2])*100)
                if dis_z >= 0.999 and dis_z <= 1.05:
                    print("[INFO] detecting '{}' markers...".format(
                        desired_aruco_dictionary))
                    print("[INFO] detected the marker:'{}'".format(marker_id))
                    sound = pyttsx.init()
                    sound.say(str(marker_id))
                    sound.runAndWait()


        ret,image = cv2.threshold(image,60,255,cv2.THRESH_BINARY)
        cv2.imshow('img' , image)
        kernel1 = numpy.ones((9,9),numpy.uint8)
        # kernel2 = numpy.ones((2,2),numpy.uint8)
        h, w, d = image.shape
                
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #hin =   cv2.warpPerspective(image, hinfo, (w,h))
        mask = cv2.inRange(hsv, lower_black, upper_black)
                
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel1)
        #mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel2)
        cv2.imshow('mask' , mask)
        
        ori = numpy.array([[8, 233], [312,233], [78, 180],[245, 180]])
        dst = numpy.array([[80, 240], [220,240], [90, 80],[220, 80]])
        # ori = numpy.array([[8, 233], [312,233], [78, 180],[245, 180]])
        # dst = numpy.array([[80, 240], [220,240], [90, 80],[220, 80]])
        # ori = numpy.array([[8, 233], [312,233], [70, 190],[250, 190]])
        # dst = numpy.array([[80, 240], [220,240], [90, 80],[220, 80]])
                        
        hinfo, status = cv2.findHomography(ori, dst)
        mask1 = cv2.warpPerspective(mask, hinfo, (w,h))
        mask2 = cv2.warpPerspective(mask, hinfo, (w,h))
        mask3 = cv2.warpPerspective(mask, hinfo, (w,h))
        mask4 = cv2.warpPerspective(mask, hinfo, (w,h))
        mask = cv2.warpPerspective(mask, hinfo, (w,h))
        detector = self.turn_detect(mask)
        mask1[0:h, w/2:w] = 0 #left
               
        # mask4[0:2*h/3, 0:w] = 0 #right
        mask4[0:h, 0:w/2] = 0 #

        mask3[0:h/3, 0:w] = 0 #right middle
        mask3[2*h/3:h, 0:w] = 0 #
        mask3[0:h, 0:w/2] = 0

        M  = cv2.moments(mask)
        M1 = cv2.moments(mask1)
                
        M3 = cv2.moments(mask3)
        M4 = cv2.moments(mask4)

        if M['m00'] > 0 and self.right==0:
            if self.init==0:
                time.sleep(1)
                self.init=1
            if self.start==0:
                self.twist.linear.x = 0.2
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(3.5)
                self.twist.linear.x = 0.2
                self.twist.angular.z = -3
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(1)
                self.twist.linear.x = 0.2
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(2.2)
                self.twist.linear.x = 0.2
                self.twist.angular.z = 2.2
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(1.2)
                self.start=1
                self.twist.linear.x = 0.15
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(1)
            thistime=time.time()
            # print(thistime-self.starttimer)
            
            cx1 = int(M['m10']/M['m00'])
            cy1 = int(M['m01']/M['m00'])
            if M1['m00']>0:
                cx2 = int(M1['m10']/M1['m00'])
                                
                cy2 = int(M1['m01']/M1['m00'])
                                
                cv2.circle(mask, (cx2, cy2), 5, (255,255,255), -1) 
                        
            if M4['m00']>0:
                #cx3 = int(M4['m10']/M4['m00'])
                cx3 = 210
                #print(cx3)
                #cy3 = int(M4['m01']/M4['m00'])
                cy3 = 134
                #print(cy3)
                cv2.circle(mask, (cx3, cy3), 5, (255,255,255), -1)
            else :
                #cx3 = 203
                cx3 = 210
                cy3 = 134
                                
            cv2.circle(mask, (cx1, cy1), 10, (255,255,255), -1)
            fx=(cx2+cx3)/2
            fy=(cy2+cy3)/2
            cv2.circle(mask, (fx, fy), 5, (255,255,255), -1)
            err = w/2 - fx
            angz=(err*90.0/160)/13
            print(angz)
            if abs(angz) > 0.15:
                self.twist.linear.x = 0.075
            else:
                self.twist.linear.x = 0.2
                # self.twist.linear.x = 0.12
                self.twist.angular.z = angz
                self.cmd_vel_pub.publish(self.twist)
                self.time = rospy.get_time()
                print(self.time-self.starttimer)
                if self.time-self.starttimer > 47:
                    print("right")
                    self.right =1
            
            # print("angular",self.twist.angular.z)
            if self.right == 1:
                self.twist.linear.x = 0.2
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(0.8)
                self.twist.linear.x = 0.2
                self.twist.angular.z = -3
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(1.2)
                self.twist.linear.x = 0.2
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(5)
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(30)

        # cv2.imshow("window", image)
        cv2.imshow("window2",mask)
        # cv2.imshow("window3",mask1)
        # cv2.imshow("window4",mask0)
        # cv2.imshow("window5",mask3)
        # cv2.imshow("window6",mask4)
        cv2.waitKey(1)

rospy.init_node('lane_follower')
follower = Follower()
rospy.spin()