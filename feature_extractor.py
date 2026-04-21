import os
import re
import math
import pandas as pd
import pymorphy3
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime

# Перевіряємо, чи завантажені необхідні мовні пакети для NLTK.
# NLTK (Natural Language Toolkit) потрібен для правильного розбиття тексту на речення та слова.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Завантаження ресурсів NLTK...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Ініціалізація морфологічного аналізатора для української мови.
# Pymorphy3 дозволяє визначати частини мови та зводити слова до початкової форми (леми).
try:
    morph = pymorphy3.MorphAnalyzer(lang='uk')
except Exception as e:
    print(f"Критична помилка ініціалізації pymorphy3: {e}")
    exit()

# Допоміжні лінгвістичні функції

def clean_text(text):
    """
    Функція очищує текст від технічних переносів та зайвих пробілів.
    
    Вхід: Сирий текст.
    Вихід: Очищений текст одним рядком.
    """
    # Видаляємо переноси слів, які часто трапляються при копіюванні з PDF (наприклад "про-\nграма")
    text = text.replace("-\n", "")
    # Замінюємо переноси рядків на пробіли
    text = text.replace("\n", " ")
    # Видаляємо подвійні пробіли
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def count_syllables_ua(word):
    """
    Підраховує кількість складів у слові для української мови.
    
    Методика: Кількість складів дорівнює кількості голосних літер.
    Голосні: а, е, є, и, і, ї, о, у, ю, я.
    """
    vowels = "аеєиіїоуюяАЕЄИІЇОУЮЯ"
    count = 0
    for char in word:
        if char in vowels:
            count += 1
    return count

def get_morphological_features(words):
    """
    Виконує морфологічний аналіз списку слів.
    
    Цей метод:
    1. Визначає частину мови для кожного слова.
    2. Рахує кількість слів кожної частини мови (іменники, дієслова тощо).
    3. Рахує кількість УНІКАЛЬНИХ лем (нормальних форм) для іменників, прикметників та дієслів.
       Це необхідно для формули читабельності Солнишкіної.
       
    Вхід: Список слів (токенів).
    Вихід: 
      - Словник з кількістю частин мови (pos_counts).
      - Кількість унікальних іменників.
      - Кількість унікальних прикметників.
      - Кількість унікальних дієслів.
    """
    # Ініціалізуємо лічильник частин мови нулями
    pos_counts = {
        'NOUN': 0, # Іменник
        'ADJF': 0, # Прикметник (повний)
        'VERB': 0, # Дієслово
        'INTJ': 0, # Вигук
        'NPRO': 0, # Займенник
        'ADVB': 0, # Прислівник
        'PRCL': 0, # Частка
        'CONJ': 0, # Сполучник
        'PREP': 0, # Прийменник
        'PRED': 0, # Предикатив
        'GRND': 0, # Дієприслівник
        'NUMR': 0, # Числівник
        'COMP': 0, # Компаратив
        'NNE': 0   # Невизначено / Інше
    }
    
    # Множини (set) зберігають тільки унікальні значення
    unique_nouns = set()
    unique_adj = set()
    unique_verbs = set()

    for word in words:
        # Аналізуємо тільки слова, ігноруємо цифри та пунктуацію
        if not word.isalpha():
            continue
            
        # Отримуємо перший (найімовірніший) варіант розбору слова
        parsed_word = morph.parse(word)[0]
        pos_tag = parsed_word.tag.POS # Частина мови
        normal_form = parsed_word.normal_form # Початкова форма слова
        
        # Розподіляємо по категоріях
        if pos_tag in pos_counts:
            pos_counts[pos_tag] += 1
        elif pos_tag == 'INFN': # Інфінітив вважаємо дієсловом
            pos_counts['VERB'] += 1
        elif pos_tag == 'ADJS': # Короткий прикметник вважаємо прикметником
            pos_counts['ADJF'] += 1
        else:
            pos_counts['NNE'] += 1
            
        # Збір унікальних слів для формули Солнишкіної
        if pos_tag == 'NOUN':
            unique_nouns.add(normal_form)
        if pos_tag in ['ADJF', 'ADJS']:
            unique_adj.add(normal_form)
        if pos_tag in ['VERB', 'INFN']:
            unique_verbs.add(normal_form)

    return pos_counts, len(unique_nouns), len(unique_adj), len(unique_verbs)

# Формули читабельності

def calc_solnyshkina_Q(ASL, ASW, unique_noun, unique_adj, unique_verb, noun_cnt, adj_cnt, verb_cnt):
    """
    Розрахунок метрики читабельності Q (Формула Солнишкіної).
    Ця формула розроблена для оцінки складності текстів навчальної літератури.
    
    Параметри:
    - ASL: Середня довжина речення.
    - ASW: Середня кількість складів у слові.
    - UNAV: Співвідношення унікальних іменників та прикметників до унікальних дієслів.
    - NAV: Співвідношення всіх іменників та прикметників до дієслів.
    """
    
    # Захист від ділення на нуль
    if unique_verb == 0:
        u_verb_safe = 1
    else:
        u_verb_safe = unique_verb
    
    if verb_cnt == 0:
        verb_safe = 1
    else:
        verb_safe = verb_cnt

    # Розрахунок коефіцієнтів лексичної різноманітності
    UNAV = (unique_noun + unique_adj) / u_verb_safe
    NAV = (noun_cnt + adj_cnt) / verb_safe
    
    # Повна регресійна формула
    # Кожен доданок зважує вплив різних лінгвістичних факторів
    part1 = (-0.124 * ASL) + (0.018 * ASW)
    part2 = (-0.007 * UNAV) + (0.007 * NAV)
    part3 = (-0.003 * (ASL ** 2)) + (0.184 * ASL * ASW)
    part4 = (0.097 * ASL * UNAV) - (0.158 * ASL * NAV)
    part5 = (0.09 * (ASW ** 2)) + (0.091 * ASW * UNAV)
    part6 = (0.023 * ASW * NAV) - (0.157 * (UNAV ** 2))
    part7 = (-0.079 * UNAV * NAV) + (0.058 * (NAV ** 2))
    
    Q = part1 + part2 + part3 + part4 + part5 + part6 + part7
    return Q

def calc_readability_metrics(stats, morph_data):
    """
    Розрахунок усіх формул читабельності.

    """
    metrics = {}
    
    # Зберігаємо базові статистики для зручності
    ASL = stats['ASL'] # Середня довжина речення
    ASW = stats['ASW'] # Середня к-сть складів
    PercComplex = stats['perc_complex_words'] # % складних слів
    
    # 1. Індекс Флеша (Flesch Reading Ease)
    # Чим вище значення, тим легше текст. Для української мови значення можуть бути нижчими за англійські.
    metrics['Flesch_RE'] = 206.835 - (1.015 * ASL) - (84.6 * ASW)
    
    # 2. Рівень Флеша-Кінкейда (Flesch-Kincaid Grade Level)
    # Показує номер класу школи США, необхідний для розуміння тексту.
    metrics['Flesch_Kincaid'] = (0.39 * ASL) + (11.8 * ASW) - 15.59

    # 3. Формула Мацковського
    metrics['Matskovskiy'] = (0.62 * ASL) + (0.123 * PercComplex) + 0.051

    # 4. Індекс Пісарека (Pisarek)
    # Для польської мови (близька структура до української).
    metrics['Pisarek'] = (ASL / 3) + (PercComplex / 3) + 1
    
    # 5. Формула Солнишкіної (Q)
    metrics['Solnyshkina'] = calc_solnyshkina_Q(
        ASL, ASW, 
        morph_data['u_noun'], morph_data['u_adj'], morph_data['u_verb'],
        morph_data['n_noun'], morph_data['n_adj'], morph_data['n_verb']
    )
    
    return metrics

# Основна логіка обробки тексту

def process_article(text):
    """
    Головна функція обробки одного тексту.
    Виконує повний цикл: очистка -> токенізація -> статистика -> морфологія -> формули.
    
    Повертає словник з усіма ознаками.
    """
    # 1. Очистка
    clean_txt = clean_text(text)
    
    # 2. Токенізація (розбиття на речення та слова)
    sentences = sent_tokenize(clean_txt)
    words_raw = word_tokenize(clean_txt)
    
    # Відбираємо тільки слова (викидаємо знаки пунктуації)
    words_alpha = []
    for w in words_raw:
        if w.isalpha():
            words_alpha.append(w)
    
    num_sentences = len(sentences)
    num_words = len(words_alpha)
    
    # Перевірка на порожній текст, щоб уникнути ділення на нуль
    if num_words == 0 or num_sentences == 0:
        return None

    # 3. Розрахунок складів та літер
    num_syllables = 0
    num_letters = 0
    complex_words_count = 0 # Складні слова (4+ склади)

    for w in words_alpha:
        syls = count_syllables_ua(w)
        num_syllables += syls
        num_letters += len(w)
        
        # Для української методики складним словом часто вважається слово з 4+ складами
        if syls >= 4:
            complex_words_count += 1
    
    # 4. Обчислення базових коефіцієнтів
    ASL = num_words / num_sentences          # Середня довжина речення (слів)
    ASW = num_syllables / num_words          # Середня к-сть складів у слові
    AvgLett = num_letters / num_words        # Середня к-сть літер у слові
    PercComplex = (complex_words_count / num_words) * 100 # Відсоток складних слів
    
    # 5. Морфологічний аналіз
    pos_counts, u_noun, u_adj, u_verb = get_morphological_features(words_raw)
    
    # 6. Збір даних для формул
    stats_for_formulas = {
        'ASL': ASL,
        'ASW': ASW,
        'perc_complex_words': PercComplex
    }
    
    morph_data_for_formulas = {
        'u_noun': u_noun, 'u_adj': u_adj, 'u_verb': u_verb,
        'n_noun': pos_counts['NOUN'], 'n_adj': pos_counts['ADJF'], 'n_verb': pos_counts['VERB']
    }
    
    # 7. Розрахунок фінальних метрик читабельності
    readability_scores = calc_readability_metrics(stats_for_formulas, morph_data_for_formulas)
    
    # 8. Формування результату
    features = {
        # Базова статистика
        "n_sentences": num_sentences,
        "n_words": num_words,
        "n_syllables": num_syllables,
        "n_complex_words": complex_words_count,
        "ASL": ASL,
        "ASW": ASW,
        "AvgLetters": AvgLett,
        
        # Морфологія (абсолютні значення)
        "n_noun": pos_counts['NOUN'],
        "n_adj": pos_counts['ADJF'],
        "n_verb": pos_counts['VERB'],
        "n_conj": pos_counts['CONJ'],
        "n_prep": pos_counts['PREP'],
        "n_pron": pos_counts['NPRO'],
        
        # Морфологія (відносні значення - частки)
        "ratio_noun": pos_counts['NOUN']/num_words,
        "ratio_verb": pos_counts['VERB']/num_words,
        "ratio_adj": pos_counts['ADJF']/num_words,
        
        # Індекси читабельності
        "Flesch_RE": readability_scores['Flesch_RE'],
        "Flesch_Kincaid": readability_scores['Flesch_Kincaid'],
        "Matskovskiy": readability_scores['Matskovskiy'],
        "Pisarek": readability_scores['Pisarek'],
        "Solnyshkina": readability_scores['Solnyshkina']
    }
    return features

# Запуск програми

def main():
    print("Початок екстракції ознак...")
    
    DATA_DIR = "data"

    # Перевірка, чи існує папка data
    if not os.path.exists(DATA_DIR):
        print(f"Помилка: Папку '{DATA_DIR}' не знайдено.")
        return
    
    # Створюємо папку для результатів з поточною датою
    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join("feature_extractor_results", f"run_{now_str}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dataset_features.csv")
    
    all_data = []
    
    # Регулярний вираз для пошуку текстів у форматі ###123### Текст...
    article_pattern = re.compile(r'###(\d+)###\s*(.*?)(?=###\d+###|\Z)', re.DOTALL)
    
    # Знаходимо всі файли, що починаються на 'f' і закінчуються на '.txt' (f1.txt, f2.txt...)
    files_found = []
    for f in os.listdir(DATA_DIR):
        if re.match(r'f\d+\.txt', f):
            files_found.append(f)
    
    # Сортуємо файли
    files_found.sort()
    
    if not files_found:
        print(f"Файли f1.txt, f2.txt... не знайдено в папці '{DATA_DIR}'.")
        return

    for filename in files_found:
        # Визначаємо рівень складності з назви файлу (f1 -> 1, f5 -> 5)
        try:
            level_match = re.search(r'f(\d+)', filename)
            if level_match:
                level = int(level_match.group(1))
            else:
                level = 0
        except:
            level = 0
            
        print(f"Обробка файлу: {filename} (Рівень {level})")
        
        #Повний шлях до файлу
        file_path = os.path.join(DATA_DIR, filename)

        # Відкриваємо файл
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Помилка читання файлу {filename}: {e}")
            continue
            
        # Знаходимо всі тексти всередині файлу
        articles = article_pattern.findall(content)
        
        for art_id, art_text in articles:
            art_text = art_text.strip()
            if not art_text:
                continue
            
            # Виклик головної функції обробки
            features = process_article(art_text)
            
            if features:
                # Формуємо рядок для таблиці
                row = {
                    "original_id": art_id,
                    "difficulty_level": level,
                    "source_file": filename,
                    "text": art_text # Зберігаємо повний текст для BERT
                }
                # Додаємо обчислені ознаки до рядка
                row.update(features)
                all_data.append(row)

    # Збереження результатів у CSV
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Переміщуємо колонку 'text' в кінець для зручності перегляду таблиці
        cols = []
        for c in df.columns:
            if c != 'text':
                cols.append(c)
        cols.append('text')
        
        df = df[cols]
        
        # Зберігаємо файл
        df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
        print(f"Дані успішно збережено у файл: {output_file}")
        
        # Виводимо приклад перших 5 рядків
        #print("Приклад даних:")
        #print(df[['difficulty_level', 'ASL', 'Solnyshkina']].head())
    else:
        print("Не вдалося зібрати дані. Перевірте вхідні файли.")

if __name__ == "__main__":
    main()