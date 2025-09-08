import random
import string
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def generate_super_resolution_chart_grid(output_filename="super_resolution_chart_grid.pdf"):
    """
    在一张A4大小的PDF上生成一个网格状的、不重叠的字符测试图表。
    字符从大到小排列，用于超分辨率测试。

    Args:
        output_filename (str): 输出的 PDF 文件名。
    """
    print(f"正在创建网格状的超分辨率测试图表: {output_filename}")

    # 获取 A4 纸张的尺寸（单位：磅）
    width, height = A4
    
    # 创建 PDF 画布
    c = canvas.Canvas(output_filename, pagesize=A4)

    # --- 定义图表参数 ---

    # 1. 字符集：大写字母和数字，更容易识别
    character_set = string.ascii_uppercase + string.digits

    # 2. 字体：使用清晰、无衬线的 Helvetica 字体
    font_name = 'Helvetica-Bold'
    
    # 3. 字体大小序列：从大到小排列
    font_sizes = [72, 60, 48, 36, 28, 24, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6, 5]

    # 4. 颜色：纯黑色，以获得最大对比度
    color = (0, 0, 0) # RGB 纯黑

    # 5. 页面边距和行间距
    margin = 20 * mm  # 20mm 边距
    line_spacing_multiplier = 1.8 # 行间距是字体大小的 1.8 倍

    # --- 开始绘制 ---

    c.setFillColorRGB(*color)

    # 设置初始绘制位置 (y坐标)
    # 从页面顶部开始，减去边距
    current_y = height - margin

    print("正在按行生成字符...")

    # 遍历每一种字体大小
    for size in font_sizes:
        # 设置当前行的字体
        c.setFont(font_name, size)
        
        # 计算当前行的行高
        line_height = size * line_spacing_multiplier
        
        # 检查如果再画一行是否会超出下边距
        if current_y - line_height < margin:
            break # 页面空间不足，停止绘制

        # 更新 y 坐标到当前行的基线
        current_y -= line_height
        
        # 设置 x 坐标从左边距开始
        current_x = margin
        
        # 在当前行内生成字符，直到超出右边距
        while current_x < width - margin:
            # 随机选择一个字符
            char_to_draw = random.choice(character_set)
            
            # 计算字符的宽度，以便确定下一个字符的位置
            # c.stringWidth 会返回给定字符串在当前字体下的宽度
            char_width = c.stringWidth(char_to_draw)
            
            # 检查如果画了这个字符是否会超出右边距
            if current_x + char_width > width - margin:
                break # 本行空间不足，换行

            # 在 (current_x, current_y) 位置绘制字符
            c.drawString(current_x, current_y, char_to_draw)
            
            # 更新 x 坐标，准备绘制下一个字符
            # 增加一个额外的间距（例如字符宽度的 1.5 倍）
            current_x += char_width * 2.5 

        print(f"  - 完成了字体大小为 {size} 磅的一行")

    # 保存 PDF 文件
    c.save()
    print(f"\n成功！图表已保存为 '{output_filename}'。")
    print("字符整齐排列，从上到下由大到小，保证不重叠。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为超分辨率实验生成一个网格状、不重叠的PDF测试图表。")
    parser.add_argument("-o", "--output", type=str, default="super_resolution_chart_grid.pdf",
                        help="输出的 PDF 文件名 (默认: super_resolution_chart_grid.pdf)")
    
    args = parser.parse_args()
    
    generate_super_resolution_chart_grid(output_filename=args.output)