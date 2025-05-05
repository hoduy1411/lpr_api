def get_max_boxs(list_boxs):
    max_s = 0
    box_max = None

    for box in list_boxs:
        x1, y1, x2, y2, _, _ = box
        s = (x2-x1) * (y2-y1)
        if s > max_s:
            max_s = s
            box_max = box
    
    return box_max


def get_max_conf(list_boxs):
    conf_max = 0
    box_max = None

    for box in list_boxs:
        conf = box[4]
        if conf > conf_max:
            conf_max = conf
            box_max = box
    
    return box_max