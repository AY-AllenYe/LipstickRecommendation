def hex_to_rgb(hex_color):
    # 去除可能的前导#号
    hex_color = hex_color.lstrip('#')
    
    # 验证长度
    if len(hex_color) != 6:
        raise ValueError(f"十六进制颜色代码必须是6位字符，当前为{len(hex_color)}位: {hex_color}")

    # 将每两位十六进制转换为十进制整数
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
        
    return r, g, b

def hex_to_hsv(hex_color):
    """
        H: 色相 [0, 360]
        S: 饱和度 [0, 1]
        V: 明度 [0, 1]
    """
    
    # 先转换为RGB
    r, g, b = hex_to_rgb(hex_color)
    
    # 将RGB值归一化到[0, 1]范围
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    # 计算最大值、最小值和差值
    cmax = max(r_norm, g_norm, b_norm)
    cmin = min(r_norm, g_norm, b_norm)
    delta = cmax - cmin
    
    # 计算明度 (Value)
    v = cmax
    
    # 计算饱和度 (Saturation)
    s = 0.0 if cmax == 0 else delta / cmax
    
    # 计算色相 (Hue)
    h = 0.0
    if delta == 0:
        h = 0.0
    elif cmax == r_norm:
        h = 60 * (((g_norm - b_norm) / delta) % 6)
    elif cmax == g_norm:
        h = 60 * (((b_norm - r_norm) / delta) + 2)
    elif cmax == b_norm:
        h = 60 * (((r_norm - g_norm) / delta) + 4)
    
    # 确保色相在[0, 360)范围内
    h = h % 360
    
    return round(h, 2), round(s, 4), round(v, 4)

# r, g, b = hex_to_rgb('#FFFFFF')
# print(r, g, b)