import numpy as np
import matplotlib.pyplot as plt

from skimage import io

import cv2
import requests

import streamlit as st

# Значение по умолчанию для URL ссылки на изображение
default_image_url = "https://timeweb.com/media/898dffddac15f940880597e2b3d2c858.jpg"

# Ввод URL ссылки на изображение
image_url = st.text_input("Введите URL ссылку на изображение", value=default_image_url)

if image_url:
    try:
        # Загрузка изображения из URL
        response = requests.get(image_url)
        image_array = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, -1)

        # Преобразование изображения в оттенки серого
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        st.image(gray_image, caption="Загруженное изображение", use_column_width=True)

        # Размер изображения = размерность матрицы
        st.write(f"Размеры фото: {gray_image.shape}")

        # Выбор количества сингулярных чисел
        singular_percent = 10
        singular_value = np.int64(gray_image.shape[0] / 100) * singular_percent
        num_singular_values = st.slider(
            "Выберите количество сингулярных чисел",
            min_value=1,
            max_value=np.min(gray_image.shape),
            value=singular_value,
        )

        # Сжатие изображения и отображение результата
        if st.button("Сжать изображение"):
            # Произведем сингулярное разложение матрицы (изображения в данном случае).
            # SVD - это метод анализа многомерных данных, который разбивает матрицу на три более простые матрицы: U, Sigma и V.
            # U (left singular vectors): Это матрица, содержащая левые сингулярные векторы.
            # sing_vals (singular values): Вектор, содержащий сингулярные значения, которые представляют важность каждого сингулярного вектора.
            # V (right singular vectors): Это матрица, содержащая правые сингулярные векторы.
            U, sing_vals, V = np.linalg.svd(gray_image)

            # Посмотрим размерность U матрицы, sing_vals вектора и V матрицы.
            # U.shape, sing_vals.shape, V.shape

            # Создаем новую матрицу sigma, которая имеет такую же форму и тип данных, как и матрица image, но все ее элементы установлены в 0.
            sigma = np.zeros_like(gray_image, dtype=np.float64)

            # Заполняем диагональ матрицы sigma значениями из вектора sing_vals.
            np.fill_diagonal(sigma, sing_vals)

            # Посмотрим размерности.
            # U.shape, sigma.shape, V.shape

            # Задаем параметр top_k, который используется для обрезки матриц до первых k столбцов/строк.

            top_k = num_singular_values

            trunc_U = U[
                :, :top_k
            ]  # trunc_U будет содержать первые top_k столбцов матрицы U.
            trunc_sigma = sigma[
                :top_k, :top_k
            ]  # trunc_sigma будет содержать верхний левый квадрат размером top_k x top_k из матрицы sigma.
            trunc_V = V[
                :top_k, :
            ]  # trunc_V будет содержать первые top_k строк матрицы V.

            # Смотрим какой процент информации в иозображении мы оставили, обрезав его параметром top_k/
            # top_k / gray_image.shape[0]

            # Посмотрим размерности.
            # trunc_U.shape, trunc_sigma.shape, trunc_V.shape

            # Восстанавливаем усеченную матрицу из сингулярного разложения.
            # trunc_image представляет собой результат умножения усеченных матриц trunc_U, trunc_sigma и trunc_V.
            # Этот шаг может быть частью процесса сжатия данных или восстановления из усеченного представления, полученного с помощью сингулярного разложения (SVD).
            # Этот процесс может быть использован для восстановления исходной матрицы, используя усеченное представление,
            # сохраняя только наиболее важные компоненты и уменьшая объем хранимых данных.
            trunc_image = trunc_U @ trunc_sigma @ trunc_V

            # Выводим оба изображения для сравнения.
            fig, ax = plt.subplots(2, 1, figsize=(15, 10))
            ax[0].imshow(gray_image, cmap="gray")
            ax[0].set_title("Исходное изображение")
            ax[1].imshow(trunc_image, cmap="gray")
            ax[1].set_title(f"Изображение на top {top_k} сингулярных чисел")

            st.pyplot(fig)

            st.write(
                f"Фото сжато на {np.round(((1 - top_k / gray_image.shape[0]) * 100), 2)} %"
            )

            # U, s, V = np.linalg.svd(img_array[:, :, 0], full_matrices=False)
            # compressed_img_r = np.dot(U[:, :num_singular_values], np.dot(np.diag(s[:num_singular_values]), V[:num_singular_values, :]))
            # U, s, V = np.linalg.svd(img_array[:, :, 1], full_matrices=False)
            # compressed_img_g = np.dot(U[:, :num_singular_values], np.dot(np.diag(s[:num_singular_values]), V[:num_singular_values, :]))
            # U, s, V = np.linalg.svd(img_array[:, :, 2], full_matrices=False)
            # compressed_img_b = np.dot(U[:, :num_singular_values], np.dot(np.diag(s[:num_singular_values]), V[:num_singular_values, :]))
            # compressed_img = np.stack([compressed_img_r, compressed_img_g, compressed_img_b], axis=-1).astype(np.uint8)

            # st.image(compressed_img, caption=f'Сжатое изображение с {num_singular_values} сингулярными числами', use_column_width=True)
    except Exception as e:
        st.write(
            "Ошибка при загрузке изображения. Пожалуйста, убедитесь, что URL ссылка корректна."
        )
