
    
    


def main(args=None):
    rclpy.init(args=args)
    
    try:
        gaze_node = EntropyGazeNode()
        rclpy.spin(gaze_node)
    except KeyboardInterrupt:
        pass
    finally:
        gaze_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


