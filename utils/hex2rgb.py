def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    
    if len(hex_color) != 6:
        raise ValueError(f"Value Error! Please enter 6 hexadecimal digits, Not {len(hex_color)} digits: {hex_color}")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
        
    return r, g, b

# r, g, b = hex_to_rgb('#FFFFFF')
# print(r, g, b)