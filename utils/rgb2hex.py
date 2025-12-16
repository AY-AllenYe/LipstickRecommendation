def rgb_to_hex(r_color, g_color, b_color):
    r = int(r_color)
    g = int(g_color)
    b = int(b_color)
    
    def judge_range(color_value):
        if color_value <= 255 and color_value >= 0:
            return True
        else:
            return False
    
    r_bool = judge_range(r)
    g_bool = judge_range(g)
    b_bool = judge_range(b)
    if r_bool and g_bool and b_bool:
        hex_string = '{:02X}_{:02X}_{:02X}'.format(r_color, g_color, b_color)
        return hex_string
    else:
        raise ValueError(f"Value Error!")


# hex = rgb_to_hex(255,255,255)
# print(hex)