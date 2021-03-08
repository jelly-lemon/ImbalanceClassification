
from compare import compare
from data import read_data

if __name__ == '__main__':
    x, y = read_data.get_data([0, 6], -1, "yeast.dat", show_info=True)

    k = 5
    while len(y[y == 0]) / k < 50:
        x, y = read_data.upsampling_copy(x, y, 1)
        print("复制一份后：%d/%d" % (len(y[y == 1]), len(y[y == 0])))

    result = compare.kFoldTest(x.copy(), y.copy(), sampler="DUS", classifier="KNN", k=k)
    print(result)