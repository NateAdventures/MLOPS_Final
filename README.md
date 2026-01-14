# Credit Default Prediction (PD-модель) — MLOps Final Project

Учебный проект по дисциплине «Автоматизация процессов разработки и тестирования моделей машинного обучения».
Цель: построить воспроизводимый end-to-end пайплайн (данные → валидация → фичи → обучение → логирование → деплой → мониторинг).

## Датасет
Default of Credit Card Clients Dataset (UCI).
Целевая переменная: `default.payment.next.month` (0/1).

## Структура проекта
- `src/` — код (подготовка данных, фичи, обучение, API)
- `tests/` — unit-тесты (pytest)
- `data/raw/` — исходные данные (большой датасет под DVC) + `sample.csv` для CI
- `data/processed/` — train/test после подготовки
- `models/` — обученная модель
- `dvc.yaml` — DVC pipeline
- `.github/workflows/ci.yml` — CI пайплайн

## Установка
Рекомендуется Python 3.9.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или .venv\\Scripts\\activate  # Windows

pip install -r requirements.txt
pip install pytest black flake8


Подготовка данных и обучение модели (DVC)

Проект использует DVC для воспроизводимости данных и моделей.

Запуск всего пайплайна:
dvc repro


Будут выполнены стадии:

prepare — валидация данных, feature engineering, split

train — обучение модели, логирование метрик и сохранение модели

Результаты:

data/processed/train.csv

data/processed/test.csv

models/credit_default_model.pkl

metrics.json

MLflow (эксперименты)

Для логирования экспериментов используется MLflow.

Запуск UI:

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000


Открыть в браузере:

http://127.0.0.1:5000


Логируются:

параметры моделей

метрики (ROC-AUC, Precision, Recall, F1)

ROC-кривая

обученная модель

Тестирование и CI

Локальный запуск тестов:

pytest -q


CI настроен через GitHub Actions и автоматически запускает:

black --check

flake8

pytest

валидацию данных с помощью Pandera

REST API (FastAPI)

Для инференса модели реализован REST API на FastAPI.

Запуск локально:

uvicorn src.api.app:app --host 0.0.0.0 --port 8000


Swagger UI:

http://127.0.0.1:8000/docs


Endpoint:

POST /predict — возвращает класс дефолта и вероятность

Мониторинг дрифта данных

Реализован простой мониторинг дрифта на основе Population Stability Index (PSI).

Скрипт имитирует поток новых данных и сравнивает их с обучающей выборкой.

Запуск:

python -m src.api.drift_monitor \
  --train data/processed/train.csv \
  --test data/processed/test.csv \
  --url http://127.0.0.1:8000/predict

Итог

В проекте реализованы:

валидация данных (Pandera)

feature engineering

обучение модели в sklearn pipeline

подбор гиперпараметров

логирование экспериментов (MLflow)

версионирование данных и моделей (DVC)

CI с тестами и линтингом

REST API для инференса

базовый мониторинг дрифта данных

Проект демонстрирует полный цикл MLOps для задачи кредитного скоринга.
