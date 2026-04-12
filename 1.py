from datasets import load_dataset
print('Загрузка датасета...')
dataset = load_dataset('code_search_net', 'python', split='train', trust_remote_code=True)

with open('codesearchnet_python.txt', 'w', encoding='utf-8') as f:
    for item in dataset:
        # Берём только код (поле 'func_code_string')
        code = item['func_code_string']
        f.write(code + '\n\n') # Разделяем файлы пустой строкой
print('Готово! Файл сохранён как codesearchnet_python.txt')