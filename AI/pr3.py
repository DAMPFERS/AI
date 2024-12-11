# import numpy as np
# import matplotlib.pyplot as plt

# def generate_and_analyze_matrix(M: int, N: int):
#     # Заполнение матрицы случайными числами, распределенными по нормальному закону
#     matrix = np.random.normal(size=(M, N))

#     # Вычисление математического ожидания и дисперсии для каждого столбца
#     mean = np.mean(matrix, axis=0)
#     variance = np.var(matrix, axis=0)

#     # Сохранение результатов в файл
#     np.savetxt('matrix.txt', matrix, delimiter=',')
#     np.savetxt('mean.txt', mean, delimiter=',')
#     np.savetxt('variance.txt', variance, delimiter=',')

#     # Построение гистограмм для каждой строки
#     for i in range(M):
#         plt.hist(matrix[i], bins=30, alpha=0.5, label=f'Row {i+1}')
#     plt.legend(loc='upper right')
#     plt.title('Histograms of each row')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.show()

#     return matrix, mean, variance



# if __name__ == "__main__":
#     # Устанавливаем количество строк и столбцов матрицы
#     M = 10  # Количество строк
#     N = 5   # Количество столбцов

#     # Вызываем функцию и получаем матрицу, математическое ожидание и дисперсию
#     matrix, mean, variance = generate_and_analyze_matrix(M, N)

#     # Выводим результаты
#     print("Matrix:")
#     print(matrix)
#     print("Mean:")
#     print(mean)
#     print("Variance:")
#     print(variance)




import numpy as np
import matplotlib.pyplot as plt

def process_matrix(file_path, output_path, M, N):
    """
    Функция создает матрицу (M, N) со случайными числами, распределенными по нормальному закону,
    считает мат. ожидание и дисперсию для каждого столбца, и строит гистограммы для каждой строки.
    
    :param file_path: путь к файлу для сохранения исходной матрицы.
    :param output_path: путь к файлу для сохранения результатов расчетов.
    :param M: количество строк в матрице.
    :param N: количество столбцов в матрице.
    """
    # Генерация матрицы
    matrix = np.random.randn(M, N)  # Генерация случайных чисел по нормальному закону
    
    # Сохранение матрицы в файл
    np.savetxt(file_path, matrix, fmt='%.5f', header='Generated matrix')
    
    # Вычисление мат. ожидания и дисперсии для каждого столбца
    mean_values = np.mean(matrix, axis=0)
    variance_values = np.var(matrix, axis=0)
    
    # Сохранение результатов в файл
    with open(output_path, 'w') as f:
        f.write("Mat. expectation by columns:\n")
        f.write(", ".join(map(str, mean_values)) + "\n\n")
        f.write("Column variance:\n")
        f.write(", ".join(map(str, variance_values)) + "\n")
    
    # Построение гистограмм для каждой строки
    for i, row in enumerate(matrix):
        plt.figure(figsize=(6, 4))
        plt.hist(row, bins=10, alpha=0.7, edgecolor='black')
        plt.title(f'Гистограмма значений строки {i+1}')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.grid(True)
        plt.savefig(f'histogram_row_{i+1}.png')
        plt.close()



if __name__ == "__main__":
    # Пример использования
    file_path = 'matrix1.txt'
    output_path = 'results1.txt'
    M, N = 5, 10  # Размер матрицы

    process_matrix(file_path, output_path, M, N)

