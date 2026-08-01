# Voice Anti-Spoofing

Проект представляет собой модель, обучаемую для определения подлинности голосовых аудиофайлов.

## Файлы
- model.py — архитектура модели
- dataset.py — загрузка данных
- train.py — обучение
- test.py — генерация CSV

## Запуск
```bash
pip install -r requirements.txt
python train.py
python test.py
python grading.py