from PIL import Image

def pil_visu(evaluate, src):
    size = 50

    img = Image.new('RGB', (size * 2 + 1, size * 2 + 1), 'white')
    pixels = img.load()

    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            res = evaluate([i, j])
            print(res)
            pixels[j+size, i+size] = (int(res[0]*100000.0)%255, int(res[1]*100000.0)%255, int(res[2]*100000.0)%255)

    img.save(src)
