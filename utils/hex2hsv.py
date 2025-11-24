def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    
    if len(hex_color) != 6:
        raise ValueError(f"Value Error! Please enter 6 hexadecimal digits, Not {len(hex_color)} digits: {hex_color}")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
        
    return r, g, b

def hex_to_hsv(hex_color):
    """
        H: Hue 色相 [0, 360]
        S: Saturation 饱和度 [0, 1]
        V: Value 明度 [0, 1]
    """
    
    r, g, b = hex_to_rgb(hex_color)
    
    # Normalize RGB to the range of [0, 1]
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    cmax = max(r_norm, g_norm, b_norm)
    cmin = min(r_norm, g_norm, b_norm)
    delta = cmax - cmin
    
    v = cmax
    s = 0.0 if cmax == 0 else delta / cmax
    h = 0.0
    if delta == 0:
        h = 0.0
    elif cmax == r_norm:
        h = 60 * (((g_norm - b_norm) / delta) % 6)
    elif cmax == g_norm:
        h = 60 * (((b_norm - r_norm) / delta) + 2)
    elif cmax == b_norm:
        h = 60 * (((r_norm - g_norm) / delta) + 4)
    
    # Ensure h (hue) is within the range of [0, 360)
    h = h % 360
    
    return round(h, 2), round(s, 4), round(v, 4)

# r, g, b = hex_to_rgb('#FFFFFF')
# print(r, g, b)