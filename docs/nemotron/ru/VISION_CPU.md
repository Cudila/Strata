# strata-vision: CPU-часть (ресайз, перекодирование), раунд 3

Изменены только `tools/vision/strata_vision.cpp` и `Vision.normalize` в `serve/server.py`.

## V1. nvidia_resize / aa_axis
- Ось, длина которой не меняется, пропускается (веса тождественного прохода ровно 0,1,0,0, результат тот же; 1920x1080 -> 1920x1088 теперь считает только вертикаль).
- Веса считаются один раз на выходную позицию, затем проход идёт по строкам в `std::thread` (потоков = max(4, hardware_concurrency/2) = 14). Арифметика та же: double, суммирование по j по возрастанию, нормировка тем же делением.
- Отладка: env `STRATA_VISION_DUMP_RGB=path` пишет ресайзнутые RGB-байты (только когда ресайз был).
- Побитовость: старый бинарник (aa_axis из прод-дерева + тот же код) и новый, по 5 картинкам раунда 2: `cmp` .sve файлов совпадает байт в байт для всех пяти; GPU-энкод детерминирован (два прогона старого тоже совпали). Размер дампа 1920x1080 = 6266880 = 1920*1088*3.

Среднее по 2-му/3-му ENC, мс (машина в этот раз быстрее, чем в ROUND2: старый resize 41/81 вместо 56/118):

| картинка | resize до | resize после | total до | total после |
|---|---|---|---|---|
| 640x480 PNG | 0 | 0 | 22.8 | 22.3 |
| 1024x768 PNG | 0 | 0 | 56.5 | 56.0 |
| 1920x1080 PNG | 41.3 | 19.5 | 240.4 | 220.5 |
| 1920x1080 JPEG | 40.9 | 23.0 | 243.1 | 224.8 |
| 2560x1440 PNG | 81.3 | 38.2 | 468.5 | 424 |

Прочие стадии (decode, tokenize, encode, write) без изменений в пределах шума. Ресайз ускорен ~2x, не больше: вертикальный проход упирается в память.

## V8. Vision.normalize -> BMP
Не-stb форматы и EXIF-повёрнутые JPEG теперь кодируются `im.save(out, format="BMP")` вместо PNG (24-битный BMP, без сжатия).
Сквозной тест через ENC (1024x768, новый strata-vision): WebP lossless, TIFF, WebP RGBA, JPEG с EXIF Orientation=6 -> `normalize` даёт `BM` (2359350 байт, 2-10 мс), ENC ответил `OK 768 ...` на все. Эмбеддинги из lossless-WebP и TIFF совпадают байт в байт с эмбеддингами исходного PNG (`cmp`); RGBA-WebP и EXIF-JPEG отличаются ожидаемо (lossy / поворот / белый фон). PNG-энкод 1024x768 в Pillow ~13 мс, от него отказались. Скрипт: scratchpad `v2/v8.py`. Замечание: BMP больше по размеру (3 байта/пиксель), но хэш-ключ кэша и temp-файл тут не критичны.

## V3-lite. Куда пишутся .sve
- `Vision.work_dir()` = `tempfile.mkdtemp` в системном temp = `/tmp`, это **tmpfs** (`df -T`: tmpfs, 63 ГБ, занято 33%); `/dev/shm` тоже tmpfs. Менять нечего. (systemd-юнит llama-swap не содержит PrivateTmp/TMPDIR.)
- dd 54 МБ на /tmp: запись 43 мс (1.3 ГБ/с), чтение 37 мс (1.5 ГБ/с; dd, одним потоком, memcpy с page cache быстрее).
- Движок логов времени read_sve/set_image_rows не пишет. По коду (`src/program/nemo_main.cpp:131`, `src/core/nemo_model.cpp:903`): fread в vector (resize обнуляет 54 МБ) + memcpy в pinned-буфер, оценка ~30-60 мс на 3311 токенов, плюс единоразовый `cudaMallocHost` при росте буфера. Оценка, не замер. Возможная микро-оптимизация (не делалась, чужой файл): читать сразу в pinned-буфер без промежуточного vector.
- Рекомендация: temp оставить как есть, `/dev/shm` не нужен.

## Не выполнено / ошибки
- `tools/nemotron/vision_check.py` не запустить: нужен torch (поставил CPU-сборку в venv scratchpad), но в `каталог чекпойнта BF16` лежат только 5 из 63 шардов: `FileNotFoundError: [Errno 2] No such file or directory: '/home/kalometik/llm/models/nemotron-super-vl/BF16/model-00061-of-00063.safetensors'`. Косинус не измерен; вместо этого эмбеддинги побайтно совпадают со старой версией (по которой раньше cos >= 0.998 уже был подтверждён).
- GPU: lock `claude-nemo-vision` взят, llama-swap остановлен на время замеров; затем `systemctl --user start llama-swap` (active) и `gpu-lock release claude-nemo-vision` выполнены.
