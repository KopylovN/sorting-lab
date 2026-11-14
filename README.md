Выполнил: Копылов Никита, Б06-505, ФБМФ МФТИ, 2026  
Рассмотрены: Insertion Sort, Selection Sort, LSD Sort  
Дополнительно реализованы стримлит и анализ памяти  

## Быстрый старт

```bash
git clone https://github.com/KopylovN/sorting-lab.git
cd sorting-lab

python -m venv .venv
source .venv/Scripts/activate        # Git Bash (Windows)
# .venv\Scripts\activate             # cmd / PowerShell
# source .venv/bin/activate          # Linux / macOS

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pytest -q                  # автотесты
python -m streamlit run app.py       # дашборд: http://localhost:8501