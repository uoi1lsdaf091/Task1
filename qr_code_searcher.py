import cv2  # Библиотека OpenCV для работы с изображениями и видео (чтение, обработка, отображение и т. д.)
from pyzbar.pyzbar import decode  # Библиотека для распознавания и декодирования QR-кодов и штрихкодов
import logging  # Встроенная библиотека Python для ведения логов (запись ошибок, сообщений и событий в файл)
import numpy as np  # Библиотека для работы с массивами и матрицами (упрощает математические операции с данными)


# Настройка логирования: записываем информацию в файл "qr_detection.log"
logging.basicConfig(
    filename="qr_detection.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Параметры
VIDEO_PATH = "образец №2.mp4"  # Путь к видеофайлу
ZOOM_FACTOR = 2  # Коэффициент увеличения изображения

# Список для хранения данных о распознанных QR-кодах
recognized_qr_codes = []

# Функция для увеличения изображения
def upscale_image(image, factor):
    logging.debug(f"Увеличение изображения с коэффициентом {factor}")
    height, width = image.shape[:2]  # Определяем высоту и ширину изображения
    # Изменяем размер изображения с использованием метода интерполяции
    resized = cv2.resize(image, (width * factor, height * factor), interpolation=cv2.INTER_CUBIC)
    return resized

# Функция для обработки одного кадра с использованием выбранного метода
def process_with_method(frame, method_name, method_function, frame_time):
    logging.debug(f"Применение метода: {method_name}")

    try:
        # Применяем метод к кадру (например, серый цвет, инверсия и т. д.)
        processed_frame = method_function(frame)
        # Увеличиваем кадр для улучшения распознавания QR-кодов
        upscaled_frame = upscale_image(processed_frame, ZOOM_FACTOR)

        # Распознаем QR-коды на кадре
        qr_codes = decode(upscaled_frame)
        logging.debug(f"Метод {method_name}: Найдено {len(qr_codes)} QR-кодов")

        for qr_code in qr_codes:
            # Извлекаем данные из QR-кода
            qr_data = qr_code.data.decode("utf-8")
            qr_polygon = qr_code.polygon
            # Преобразуем координаты обратно для оригинального размера кадра
            coordinates = [(int(point.x / ZOOM_FACTOR), int(point.y / ZOOM_FACTOR)) for point in qr_polygon]

            # Если есть координаты, рисуем бокс вокруг QR-кода
            if len(coordinates) >= 4:
                pts = np.array(coordinates, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # Зеленый бокс

            logging.info(
                f"Метод {method_name}: Распознан QR-код с содержимым '{qr_data}' на {frame_time:.2f} сек. Координаты: {coordinates}"
            )

            # Проверяем, был ли этот QR-код уже добавлен
            if not any(entry['data'] == qr_data and entry['coordinates'] == coordinates for entry in recognized_qr_codes):
                recognized_qr_codes.append({
                    "time": frame_time,
                    "data": qr_data,
                    "method": method_name,
                    "coordinates": coordinates
                })

            print(f"Распознан QR-код: '{qr_data}'")  # Выводим содержимое QR-кода

        return len(qr_codes) > 0  # Если QR-коды найдены, возвращаем True

    except Exception as e:
        logging.error(f"Ошибка в методе {method_name}: {e}")
        return False

# Функция для обработки одного кадра
def process_frame(frame, frame_time):
    logging.info(f"Обработка кадра на {frame_time:.2f} сек начата")

    # Список методов для обработки кадра
    methods = {
        "Оригинал": lambda x: x,  # Без изменений
        #"Инверсия": lambda x: cv2.bitwise_not(x),  # Инверсия цветов
        #"Серый": lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)  # Преобразование в оттенки серого
    }

    found_needed_qr = False  # Флаг, найден ли QR-код

    for method_name, method_function in methods.items():
        # Применяем все методы к кадру
        found_needed_qr = process_with_method(frame, method_name, method_function, frame_time)

    if not found_needed_qr:
        logging.warning(f"Нужный QR-код не найден на {frame_time:.2f} сек")

    logging.info(f"Обработка кадра на {frame_time:.2f} сек завершена")
    return frame

# Основная функция обработки видео
def process_video():
    logging.info(f"Начата обработка видео: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)  # Открываем видеофайл
    fps = cap.get(cv2.CAP_PROP_FPS)  # Получаем FPS видео

    if not cap.isOpened():
        logging.error("Не удалось открыть видео.")
        print("Не удалось открыть видео.")
        return

    while True:
        ret, frame = cap.read()  # Читаем кадр
        if not ret:  # Если кадры закончились, выходим
            break

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Текущий номер кадра
        frame_time = current_frame / fps  # Вычисляем время кадра в секундах

        frame = process_frame(frame, frame_time)  # Обрабатываем кадр

        # Показываем кадр в окне
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Останавливаем обработку, если нажата клавиша 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Обработка видео завершена")

    # Выводим результаты
    print("Распознанные QR-коды:")
    for entry in recognized_qr_codes:
        print(f"[ {entry['time']:.2f} сек ] Содержание: {entry['data']}. Метод: {entry['method']}. Координаты: {entry['coordinates']}")

# Функция для захвата видео с веб-камеры
def process_camera():
    logging.info("Начат захват с веб-камеры")
    cap = cv2.VideoCapture(0)  # Открываем веб-камеру

    if not cap.isOpened():
        logging.error("Не удалось открыть веб-камеру.")
        print("Не удалось открыть веб-камеру.")
        return

    while True:
        ret, frame = cap.read()  # Читаем кадр с камеры
        if not ret:
            logging.error("Ошибка чтения кадра с веб-камеры.")
            break

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Время кадра в секундах
        frame = process_frame(frame, frame_time)  # Обрабатываем кадр

        cv2.imshow("Camera", frame)  # Показываем кадр

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Останавливаем обработку, если нажата клавиша 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Захват с веб-камеры завершён")

if __name__ == "__main__":
    # Режим работы программы
    mode = input("Выберите режим (1 - Видео, 2 - Веб-камера): ")
    if mode == "1":
        process_video()
    elif mode == "2":
        process_camera()
    else:
        print("Неверный выбор режима.")