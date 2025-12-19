def is_accident(prev_speed, curr_speed, iou):
    SPEED_DROP_THRESHOLD = 5    # pixels/frame
    IOU_THRESHOLD = 0.15

    speed_drop = prev_speed - curr_speed

    return speed_drop > SPEED_DROP_THRESHOLD and iou > IOU_THRESHOLD
