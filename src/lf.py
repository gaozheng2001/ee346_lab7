#!/usr/bin/env python

import rospy
import cv2
import cv_bridge
import numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class Follower:

    def __init__(self):

        self.bridge = cv_bridge.CvBridge()

        self.image_sub = rospy.Subscriber('camera/image_projected_compensated',
                                          Image, self.image_callback)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                                           Twist, queue_size=10)
        self. start = 0
        self.init = 0
        self.twist = Twist()

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

        # import pdb;pdb.set_trace()
        cameraMatrix = numpy.matrix(numpy.array([[-3.313061098815712, 0.0, 160.5],
                                                 [0.0, -3.313061098815712, 120.5],
                                                 [0.0, 0.0, 1.0]]))
        distCoeffs = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0])
        R = numpy.matrix(numpy.array([[1, 0, 1],
                                      [0, 0, 0],
                                      [0, -1, 0]]))
        T = numpy.matrix(numpy.array([0, 1, 1]).reshape(3, 1))
        n = numpy.matrix(numpy.array([0, -1, 0]).reshape(3, 1))
        d = 0.115
        #H = numpy.dot(K, numpy.dot(R - numpy.dot(T, n.reshape(1, 3))/d, K.I))

        this_aruco_dictionary = cv2.aruco.Dictionary_get(
            ARUCO_DICT[desired_aruco_dictionary])
        this_aruco_parameters = cv2.aruco.DetectorParameters_create()

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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

                    self.twist.linear.x = 0
                    self.twist.angular.z = 0
                    self.cmd_vel_pub.publish(self.twist)
                    import time
                    time.sleep(10)

        lower_yellow = numpy.array([0, 0, 0])
        upper_yellow = numpy.array([0, 0, 45])

        lower_white = numpy.array([0, 0, 0])
        upper_white = numpy.array([0, 0, 45])

        mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask2 = cv2.inRange(hsv, lower_white, upper_white)

        h, w, d = image.shape
        '''
        #src_list = [[33, 229], [279, 229], [186, 144], [131, 145]]
        src_list = [[100, 163], [189, 162], [4, 239], [264, 239]]
        #dst_list = [[110, 230], [210, 230], [210, 5], [110, 5]]
        dst_list = [[110, 5], [210, 5], [110, 235], [210, 235]]
        H, _ = cv2.findHomography(numpy.array(src_list).reshape(-1, 1, 2),
                                  numpy.array(dst_list).reshape(-1, 1, 2), cv2.RANSAC, 5.0)
        '''
        #image_homography = cv2.warpPerspective(image, H, (w, h))
        #hsv_homo = cv2.cvtColor(image_homography, cv2.COLOR_BGR2HSV)
        #hmask1 = cv2.inRange(hsv_homo, lower_yellow, upper_yellow)
        #hmask2 = cv2.inRange(hsv_homo, lower_white, upper_white)

        search_top = int(2*h/3)
        # search_line = int(w/2)
        mask1[0:search_top, 0:w] = 0
        # mask1[0:h, search_line:w] = 0
        mask2[0:search_top, 0:w] = 0

        #hmask1[0:1*h/3, 0:w] = 0
        #hmask2[0:1*h/3, 0:w] = 0

        # mask2[0:h, 0:search_line] = 0

        M1 = cv2.moments(mask1)
        M2 = cv2.moments(mask2)

        #hM1 = cv2.moments(hmask1)
        #hM2 = cv2.moments(hmask2)

        def center(M):
            return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

        if M1['m00'] > 0:
            
            cx1, cy1 = center(M1)
            cx2, cy2 = center(M2)

            #hcx1, hcy1 = center(hM1)
            #hcx2, hcy2 = center(hM2)

            # import pdb;pdb.set_trace()
            fpt_x = (cx1 + cx2)/2
            fpt_y = (cy1 + cy2)/2 + 2*h/3

            #hfpt_x = (hcx1 + hcx2)/2
            #hfpt_y = (hcy1 + hcy2)/2 + 2*h/3

            cv2.circle(image, (cx1, cy1), 10, (0, 255, 255), -1)
            cv2.circle(image, (cx2, cy2), 10, (255, 255, 255), -1)
            cv2.circle(image, (fpt_x, fpt_y), 10, (128, 128, 128), -1)

            #cv2.circle(image_homography, (hcx1, hcy1), 10, (0, 255, 255), -1)
            #cv2.circle(image_homography, (hcx2, hcy2), 10, (255, 255, 255), -1)
            # cv2.circle(image_homography, (hfpt_x, hfpt_y),
            #           10, (128, 128, 128), -1)

            err = w/2 - fpt_x
            #herr = w/2 - hfpt_x

            self.twist.linear.x = 0.3
            self.twist.angular.z = (err*90.0/160)/5
            self.cmd_vel_pub.publish(self.twist)
        cv2.imshow("window", image)
        cv2.waitKey(1)
        #cv2.imshow("Homography", image_homography)
        # cv2.waitKey(1)
        # import pdb;pdb.set_trace()


rospy.init_node('lane_follower')
follower = Follower()
rospy.spin()
