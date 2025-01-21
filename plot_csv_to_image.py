import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_csv_to_image(csv_file, output_image):
    try:
        # Чтение данных из CSV файла
        data = pd.read_csv(csv_file)

        # Определение типа графика на основе столбцов
        if 'T' in data.columns and 'Duration' in data.columns:
            # Построение линейного графика
            plt.figure(figsize=(10, 6))
            plt.plot(data['T'], data['Duration'], marker='o', linestyle='-', color='b')
            plt.title('График T vs Duration', fontsize=14)
            plt.xlabel('T', fontsize=12)
            plt.ylabel('Duration', fontsize=12)
            plt.grid(True)

        elif 'Default' in data.columns and 'SIMD' in data.columns:
            # Проверка количества строк
            if len(data) != 1:
                raise ValueError("CSV should contain only 1 row.")

            # Построение гистограммы
            values = data.iloc[0]
            categories = ['Default', 'SIMD']
            plt.figure(figsize=(8, 6))
            plt.bar(categories, values, color=['blue', 'green'])
            plt.title('Гистограмма Default vs SIMD', fontsize=14)
            plt.ylabel('Значения', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        else:
            raise ValueError("CSV should contain 'T' and 'Duration', or 'Default' and 'SIMD' columns.")

        # Сохранение графика в файл
        plt.savefig(output_image)
        print(f"Image saved to: {output_image}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python script.py <input_csv> <output_image>")
    else:
        input_csv = sys.argv[1]
        output_image = sys.argv[2]
        plot_csv_to_image(input_csv, output_image)