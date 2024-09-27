import argparse
import sys
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('file1')
parser.add_argument('file2')
parser.add_argument('-r', '--recursive', action='store_true')

args = parser.parse_args()

def main():
    image1 = open_image(args.file1)
    image2 = open_image(args.file2)

    if not image1:
        sys.stderr.write(f"Can't open {args.file1}\n")
        return
    elif not image2:
        sys.stderr.write(f"Can't open {args.file2}\n")
        return

    diff = imgdiff(image1, image2) 
    print(f"diff: {diff:.4f}")


def open_image(file:str, recursive:bool=False) -> Image:
    try:
        return Image.open(file)
    except:
        return []

def imgdiff(image1:Image, image2:Image):
    if image1.size != image2.size:
        sys.stderr.write('Image size is different')

    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # 픽셀 단위 비교
    difference = np.abs(image1_array - image2_array)

    # 차이가 있는 픽셀의 총합 계산 (유사도 측정)
    total_difference = np.sum(difference)

    # 유사도 계산 (0에 가까울수록 이미지가 유사함)
    similarity = (total_difference / (image1_array.size * 255))

    return similarity

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Inturrupt")
        exit()