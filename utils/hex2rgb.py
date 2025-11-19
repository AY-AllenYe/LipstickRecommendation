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

# r, g, b = hex_to_rgb('#FFFFFF')
# print(r, g, b)