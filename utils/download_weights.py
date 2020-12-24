import gdown

url = 'https://drive.google.com/uc?id=1_Uo_jB4ZVsBRA7I3MLisq_GZ8EQHeWr9'
output = 'fishv4/fish.weights'
gdown.download(url, output, quiet=False)