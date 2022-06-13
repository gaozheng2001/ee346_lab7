#!/usr/bin/env python
import rospy, cv2, cv_bridge, numpy, pyttsx
from sensor_msgs.msg import CompressedImage

class ARcodeDetector:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber('camera/image/compressed',
                CompressedImage, self.image_callback)

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
                print(dis_z)
                #if dis_z >= 80 and dis_z <= 117:
                print("[INFO] detecting '{}' markers...".format(
                    desired_aruco_dictionary))
                print("[INFO] detected the marker:'{}'".format(marker_id))
                sound = pyttsx.init()
                sound.say(str(marker_id))
                sound.runAndWait()

        cv2.imshow("window3",image)
        cv2.waitKey(1)

rospy.init_node('ARUCO_Detector')
follower = ARcodeDetector()
rospy.spin()