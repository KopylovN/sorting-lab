Выполнил: Копылов Никита, Б06-505, ФБМФ МФТИ, 2026  
Рассмотрены: Insertion Sort, Selection Sort, LSD Sort  
Дополнительно реализованы стримлит и анализ памяти  

## Быстрый старт

```bash
git clone <https://github.com/KopylovN/sorting-lab.git>
cd sorting-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -q                        # автотесты
streamlit run app.py             # дашборд: http://localhost:8501

# полный воспроизводимый прогон
python -m bench.cli --profile full --repeats 7 --memory