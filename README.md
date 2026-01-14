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
