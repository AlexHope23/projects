"""Matrix class"""
import random
import copy
import numpy as np

class Matrix:
    """Class representing matrix functions"""
    def __init__(self, nrows, ncols, init = 'zeros'):
        """Конструктор класса Matrix.
        Создаёт матрицу резмера nrows x ncols и инициализирует её методом init.
        nrows - количество строк матрицы
        ncols - количество столбцов матрицы
        init - метод инициализации элементов матрицы:
            "zeros" - инициализация нулями
            "ones" - инициализация единицами
            "random" - случайная инициализация
            "eye" - матрица с единицами на главной диагонали
        """
        if nrows < 0 or ncols < 0:
            raise ValueError
        if init not in ['zeros', 'ones', 'eye', 'random']:
            raise ValueError
        self.nrows = nrows
        self.ncols = ncols
        self.data = [[]]
        if init == "zeros":
            self.data = [[0] * ncols for _ in range(nrows)]
        elif init == "ones":
            self.data = [[1] * ncols for _ in range(nrows)]
        elif init == "eye":
            if nrows != ncols:
                raise ValueError
            self.data = [[1 if i == j else 0 for j in range(ncols)] for i in range(nrows)]
        elif init == "random":
            self.data = [[random.random() for _ in range(ncols)] for _ in range(nrows)]
    @staticmethod
    def from_dict(data):
        "Десериализация матрицы из словаря"
        ncols = data["ncols"]
        nrows = data["nrows"]
        items = data["data"]
        assert len(items) == ncols*nrows
        result = Matrix(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                result[(row, col)] = items[ncols*row + col]
        return result
    @staticmethod
    def to_dict(matr):
        "Сериализация матрицы в словарь"
        assert isinstance(matr, Matrix)
        nrows, ncols = matr.shape()
        data = []
        for row in range(nrows):
            for col in range(ncols):
                data.append(matr[(row, col)])
        return {"nrows": nrows, "ncols": ncols, "data": data}
    def __str__(self):
        matr_str = ''
        for row in self.data:
            matr_str += " ".join(str(val) for val in row) + '\n'
        return matr_str
    def __repr__(self):
        return f"Matrix({self.nrows}, {self.ncols}, data={self.data})"
    def shape(self):
        """возвращает размер массива"""
        return (self.nrows, self.ncols)
    def __getitem__(self, index):
        """возвращает элемент матрицы по индексу"""
        if not (isinstance(index, (list, tuple))) or len(index) != 2:
            raise ValueError
        row, col = index
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError
        return self.data[row][col]
    def __setitem__(self, index, value):
        """присваивает элементу матрицы новое значение"""
        if not (isinstance(index, (list, tuple))) or len(index) != 2:
            raise ValueError
        row, col = index
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError
        self.data[row][col] = value
    def __sub__(self, rhs):
        """вычитание"""
        if self.shape() != rhs.shape():
            raise ValueError
        result = Matrix(self.nrows, self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):
                result[(row, col)] = self[(row, col)] - rhs[(row, col)]
        return result
    def __add__(self, rhs):
        """сложение"""
        if self.shape() != rhs.shape():
            raise ValueError
        result = Matrix(self.nrows, self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):
                result[(row, col)] = self[(row, col)] + rhs[(row, col)]
        return result
    def __mul__(self, rhs):
        """умножение"""
        if self.ncols != rhs.nrows:
            raise ValueError
        result = Matrix(self.nrows, rhs.ncols)
        for row in range(self.nrows):
            for col in range(rhs.ncols):
                result[(row, col)] = sum(self[(row, k)] * rhs[(k, col)] for k in range(self.ncols))
        return result
    def __pow__(self, power):
        """возведение всех элементов матрицы в степень"""
        result = Matrix(self.nrows, self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):
                result[(row, col)] = self[(row, col)]**power
        return result
    def sum(self):
        """сумма всех элементов матрицы"""
        s = 0
        for row in range(self.nrows):
            for col in range(self.ncols):
                s += self[row, col]
        return s
    def det(self):
        """определитель матрицы"""
        if self.nrows == 1 and self.ncols == 1:
            return self[0, 0]
        determinant = 0
        for j in range(self.ncols):
            submatrix = self.submatrix(0, j)
            determinant += (-1) ** j * self[0, j] * submatrix.det()
        return determinant
    def submatrix(self, i, j):
        """вспомогательная функция, считающая миноры"""
        submatrix = Matrix(self.ncols-1, self.nrows - 1)
        sub_row = 0
        for row in range(self.nrows):
            if row == i:
                continue
            sub_col = 0
            for col in range(self.ncols):
                if col == j:
                    continue
                submatrix[sub_row, sub_col] = self[row, col]
                sub_col += 1
            sub_row += 1
        return submatrix
    def transpose(self):
        "Транспонирование матрицы"
        transposed = Matrix(self.ncols, self.nrows)
        for i in range(self.nrows):
            for j in range(self.ncols):
                transposed[j, i] = self[i, j]
        return transposed
    def cofactor_matrix(self):
        """матрица алгебраических дополнений"""
        cofactors = Matrix(self.nrows, self.ncols)
        for i in range(self.nrows):
            for j in range(self.ncols):
                submatrix = self.submatrix(i, j)
                cofactors[i, j] = ((-1) ** (i + j)) * submatrix.det()
        return cofactors
    def inv(self):
        """обратная матрица"""
        if self.nrows != self.ncols:
            raise ArithmeticError
        if self.det() == 0:
            raise ArithmeticError
        cofactors = self.cofactor_matrix()
        adjoint = cofactors.transpose()
        for i in range(self.nrows):
            for j in range(self.ncols):
                adjoint[i, j] /= self.det()
        return adjoint
    def tonumpy(self):
        """преобразование в массив numpy"""
        return np.array(self.data)
def test1():
    """проверка на совпадение с numpy"""
    matr1 = Matrix(3, 4, init = 'random')
    matr2 = copy.copy(matr1)
    matr2 = matr2.tonumpy()
    matr3 = Matrix(3, 4, init = 'random')
    transpose1 = matr3.transpose()
    matr3 = matr3.tonumpy()
    transpose2 = np.transpose(matr3)
    mul2 = np.dot(matr2, transpose2)
    mul1 = matr1*transpose1
    det1 = mul1.det()
    inv1 = mul1.inv()
    pow1 = inv1**3
    sum1 = pow1.sum()
    det2 = np.linalg.det(mul2)
    inv2 = np.linalg.inv(mul2)
    pow2 = np.power(inv2, 3)
    sum2 = np.sum(pow2)
    mul1 = mul1.tonumpy()
    inv1 = inv1.tonumpy()
    pow1 = pow1.tonumpy()
    assert np.array_equal(transpose1.tonumpy(), transpose2), "Транспонирование матриц не совпадает"
    assert np.array_equal(np.round(mul1,8), np.round(mul2,8)), "Произведение матриц не совпадает"
    assert round(det1,8) == round(det2,8),  "Определители не совпадают"
    assert np.array_equal(np.round(inv1,8), np.round(inv2,8)), "Обратные матрицы не совпадают"
    assert np.array_equal(np.round(pow1,6), np.round(pow2,6)), "Возведение в степень не совпадает"
    assert round(sum1,6) == round(sum2,6), "Суммы всех элементов матрицы не совпадают"
def test2():
    """продолжение"""
    matr1 = Matrix(3, 4, init = 'random')
    matr2 = copy.copy(matr1)
    matr2 = matr2.tonumpy()
    matr3 = Matrix(3, 4, init = 'random')
    shape1 = matr1.shape()
    shape2 = np.shape(matr2)
    assert shape1 == shape2, "Размеры не совпадают"
    sub1 = matr1 - matr3
    sub1 = sub1.tonumpy()
    add1 = matr1 + matr3
    add1 = add1.tonumpy()
    matr3 = matr3.tonumpy()
    sub2 = matr2 - matr3
    add2 = matr2 + matr3
    assert np.array_equal(sub1, sub2), "Разница матриц не совпадает"
    assert np.array_equal(add1, add2), "Сумма матриц не совпадает"
    getitem1 = matr1[(2,2)]
    getitem2 = matr2.item((2,2))
    assert getitem1 == getitem2, "Элементы не совпадают"
    matr1[(2,2)] = 8
    matr1 = matr1.tonumpy()
    matr2.itemset((2,2), 8)
    assert np.array_equal(matr1, matr2), "Матрицы после замены элемента не совпадают"
    print("Все тесты пройдены успешно")
if __name__ == "__main__":
    test1()
    test2()
    