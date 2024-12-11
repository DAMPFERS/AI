def prime_factors(n: int) -> list:
    '''
     разлагает число на простые множители и возвращает список этих множителей.
    ''' 
    i = 2                   # Начинаем с наименьшего простого числа
    factors = []            # Список для хранения простых множителей
    while i * i <= n:       # Пока i в квадрате меньше или равно n
        if n % i:           # Если n не делится на i
            i += 1          # Увеличиваем i на 1
        else:               # Если n делится на i
            n //= i         # Делим n на i
            factors.append(i)   # Добавляем i в список множителей
    if n > 1:                   # Если n больше 1, значит оно само является простым числом
        factors.append(n)       # Добавляем n в список множителей
    return factors              # Возвращаем список множителей



def format_factors(factors: list) -> str:
    '''
    принимает список множителей и форматирует их в строку в указанном формате
    ''' 
    
    factor_count = {}                       # Словарь для подсчета количества каждого множителя
    for factor in factors:                  # Проходим по каждому множителю
        if factor in factor_count:          # Если множитель уже в словаре
            factor_count[factor] += 1       # Увеличиваем его счетчик
        else:                               # Если множителя нет в словаре
            factor_count[factor] = 1        # Добавляем его со счетчиком 1

    result = []                                 # Список для хранения отформатированных множителей
    for factor, count in factor_count.items():  # Проходим по каждому множителю и его счетчику
        if count == 1:                          # Если счетчик равен 1                          
            result.append(f"({factor})")        # Добавляем множитель в формате (factor)
        else:                                   # Если счетчик больше 1
            result.append(f"({factor}**{count})")   # Добавляем множитель в формате (factor**count)

    return ''.join(result)                          # Объединяем все элементы списка в одну строку и возвращаем её



def factorize(n):
    factors = prime_factors(n)
    return format_factors(factors)


if __name__ == "__main__":
    numbers = (5,86240, 1247972, 100, 4628453, 551, 144, 1234525386396, 85546)
    for i in numbers:
        print(f"num = {i}: {factorize(i)}")
