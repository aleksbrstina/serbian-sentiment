# Specifikacija projekta: Klasifikacija sentimenta recenzija na srpskom jeziku

Nov 26, 2026 · @Aleksandra

## Uvod

Projekat razvija sistem za klasifikaciju sentimenta recenzija na srpskom jeziku i upoređuje tri pristupa iste složenosti problema, klasični ML model, neuronsku mrežu napisanu od nule i fino podešeni transformer. Rezultat je javni GitHub repozitorijum sa reproduktivnim eksperimentima, izveštajem o analizi grešaka i interaktivnim demo prikazom.

Cilj projekta je da u CV-u demonstrira zahteve pozicije Applied Scientist Student Worker u preduzeću Microsoft. Oglas traži sposobnost izvlačenja uvida iz podataka, razumevanje klasičnih ML algoritama, neuronskih mreža i transformera, kao i pripremu podataka i iterativno eksperimentisanje.

U obim projekta ulaze binarna klasifikacija sentimenta (pozitivno i negativno), opciono trostepena klasifikacija (pozitivno, neutralno, negativno), eksplorativna analiza podataka, poređenje modela i demo aplikacija. Van obima su prikupljanje podataka u produkcionom obimu, višejezična podrška i trening modela od nule na nivou pretreniranja.

## Funkcionalni zahtevi

Sistem mora da ispuni sledeće zahteve, grupisane po fazama obrade podataka.

| ID | Zahtev | Prioritet |
| --- | --- | --- |
| FR-01 | Sistem učitava javni skup recenzija na srpskom jeziku (SerbMR ili SentiComments.SR) iz lokalnih fajlova u jedinstveni format (tekst, oznaka, izvor). | Obavezno |
| FR-02 | Sistem normalizuje tekst prevođenjem ćirilice u latinicu, ujednačavanjem dijakritika, uklanjanjem HTML ostataka i normalizacijom razmaka. | Obavezno |
| FR-03 | Sistem uklanja duplikate i prazne zapise i beleži broj uklonjenih zapisa po kategoriji u izveštaj o kvalitetu podataka. | Obavezno |
| FR-04 | Sistem deli podatke na trening, validacioni i test skup (80/10/10) stratifikovano po oznaci, sa fiksnim seed-om. | Obavezno |
| FR-05 | Sistem generiše eksplorativnu analizu sa distribucijom klasa, distribucijom dužine teksta i najinformativnijim rečima po klasi. | Obavezno |
| FR-06 | Sistem trenira klasične modele (TF-IDF sa logističkom regresijom, linearnim SVM-om i Naive Bayes klasifikatorom) uz pretragu hiperparametara na validacionom skupu. | Obavezno |
| FR-07 | Sistem trenira neuronsku mrežu (BiLSTM nad embedinzima) implementiranu u PyTorch-u bez gotovih trening wrapper-a. | Obavezno |
| FR-08 | Sistem fino podešava transformer model (BERTić, rezervna opcija XLM-RoBERTa base) preko Hugging Face biblioteke. | Obavezno |
| FR-09 | Sistem evaluira sve modele na istom test skupu metrikama accuracy, macro F1, precision, recall i matricom konfuzije. | Obavezno |
| FR-10 | Sistem beleži vreme treninga, vreme inferencije po primeru i veličinu modela za svaki model. | Obavezno |
| FR-11 | Sistem izdvaja pogrešno klasifikovane primere i kategoriše ih (negacija, ironija, sleng, mešoviti sentiment, kratak tekst). | Obavezno |
| FR-12 | Sistem generiše zbirni izveštaj (tabela i grafikoni) poređenja modela. | Obavezno |
| FR-13 | Demo aplikacija prima tekst i prikazuje predikciju i verovatnoću sva tri modela jedno pored drugog. | Obavezno |
| FR-14 | Demo aplikacija prikazuje objašnjenje predikcije klasičnog modela (reči sa najvećim doprinosom). | Poželjno |
| FR-15 | Sistem podržava trostepenu klasifikaciju ukoliko skup podataka sadrži neutralnu klasu. | Poželjno |
| FR-16 | Sistem prikazuje mape pažnje (attention) transformer modela za izabrane primere. | Poželjno |

## Nefunkcionalni zahtevi

| ID | Zahtev |
| --- | --- |
| NFR-01 | Reproduktivnost. Svi eksperimenti koriste fiksni seed, a svako pokretanje se konfiguriše YAML fajlom koji se čuva zajedno sa rezultatima. |
| NFR-02 | Hardver. Klasični modeli i BiLSTM se treniraju na CPU-u za manje od 15 minuta. Transformer se trenira na besplatnom GPU-u (Google Colab ili Kaggle) za manje od jednog sata. |
| NFR-03 | Kvalitet koda. Kod je modularan, sa type hint-ovima i docstring-ovima, formatiran alatom ruff i pokriven unit testovima za obradu podataka i metrike. |
| NFR-04 | Praćenje eksperimenata. Parametri, metrike i artefakti svakog pokretanja beleže se u MLflow (lokalno). |
| NFR-05 | Dokumentacija. README na engleskom jeziku sadrži opis problema, tabelu rezultata, uputstvo za pokretanje i link ka demo aplikaciji. |
| NFR-06 | Performanse demo aplikacije. Odgovor za jedan tekst stiže za manje od 2 sekunde na CPU-u. |
| NFR-07 | Licence. Korišćeni skupovi podataka i modeli dozvoljavaju nekomercijalnu i istraživačku upotrebu, a izvori se navode u README fajlu. |
| NFR-08 | Veličina repozitorijuma. Podaci i težine modela se ne čuvaju u git-u. Težine transformera se objavljuju na Hugging Face Hub-u. |

## Funkcionalnosti po modulima

Sistem se sastoji od sedam modula, od kojih svaki ima jasno definisan ulaz i izlaz.

1. **Modul za podatke (`data`).** Modul učitava sirove fajlove, sprovodi normalizaciju teksta, uklanja duplikate i deli skup. Izlaz su fajlovi `train.parquet`, `val.parquet` i `test.parquet` u direktorijumu `data/processed` i izveštaj `data_quality.md` sa opisom svakog koraka čišćenja i brojem uklonjenih zapisa.
2. **Modul za eksplorativnu analizu (`eda`).** Modul generiše grafikone distribucije klasa i dužine teksta, listu najfrekventnijih i najinformativnijih reči po klasi (chi-kvadrat test) i kratak tekstualni pregled uvida. Izlaz se čuva u `reports/eda`, a isti sadržaj prikazuje i notebook `01_eda.ipynb`.
3. **Modul za klasične modele (`models/classical`).** Modul gradi scikit-learn Pipeline (TfidfVectorizer sa unigramima i bigramima reči i karakterskim n-gramima, zatim klasifikator). Pretraga hiperparametara (GridSearchCV ili Optuna) koristi macro F1 na validacionom skupu. Modul izvozi i koeficijente modela radi interpretacije.
4. **Modul za neuronsku mrežu (`models/neural`).** Modul sadrži sopstveni tokenizer i rečnik, PyTorch Dataset i DataLoader, BiLSTM arhitekturu (embedding sloj inicijalizovan fastText vektorima za srpski jezik, dvosmerni LSTM, dropout, linearni izlaz) i ručno napisanu trening petlju sa early stopping-om i čuvanjem najboljeg checkpoint-a.
5. **Modul za transformer (`models/transformer`).** Modul fino podešava model `classla/bcms-bertic` kroz Hugging Face Trainer, sa maksimalnom dužinom sekvence 256, linearnim warmup rasporedom učenja i evaluacijom posle svake epohe. Skripta mora da radi i lokalno i u Colab okruženju.
6. **Modul za evaluaciju (`evaluation`).** Modul učitava sve sačuvane modele, računa metrike na test skupu, meri vreme inferencije, generiše matrice konfuzije i zbirnu tabelu `reports/comparison.md`. Poseban deo izdvaja greške, označava ih heuristikama (prisustvo negacije "ne", "nije", "nisam", dužina teksta, upitnici i znaci uzvika) i čuva primere za ručnu analizu.
7. **Demo aplikacija (`app`).** Gradio aplikacija prima tekst i prikazuje oznaku i verovatnoću za sva tri modela, uz istaknute reči koje najviše doprinose odluci klasičnog modela. Aplikacija se objavljuje na Hugging Face Spaces.

## Arhitektura i struktura repozitorijuma

Arhitektura je linearni pipeline u kome svaki korak čita izlaz prethodnog sa diska, pa se svaki korak može pokrenuti i testirati nezavisno. Tri modela dele isti obrađeni skup i isti modul za evaluaciju, što garantuje fer poređenje.

&#91;embedded content: arhitektura sistema · sedam modula\]

Evaluacija i demo aplikacija pristupaju modelima isključivo preko interfejsa Predictor, pa se sva tri modela porede pod istim uslovima.

Struktura repozitorijuma je sledeća.

```text
serbian-sentiment/
├── configs/                 # YAML konfiguracije po modelu
│   ├── classical.yaml
│   ├── bilstm.yaml
│   └── bertic.yaml
├── data/                    # nije u git-u
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── colab_train_bertic.ipynb
├── src/sentiment/
│   ├── data/                # loading.py, cleaning.py, split.py
│   ├── eda/                 # stats.py, plots.py
│   ├── models/
│   │   ├── classical.py
│   │   ├── neural/          # tokenizer.py, dataset.py, model.py, train.py
│   │   └── transformer.py
│   ├── evaluation/          # metrics.py, errors.py, report.py
│   ├── predict.py           # jedinstven interfejs za sva tri modela
│   └── utils.py             # seed, logovanje, učitavanje konfiguracije
├── scripts/                 # CLI ulazne tačke
│   ├── prepare_data.py
│   ├── train.py             # --config configs/<model>.yaml
│   └── evaluate.py
├── app/app.py               # Gradio demo
├── reports/                 # generisani izveštaji i grafikoni
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

Svi modeli implementiraju zajednički interfejs `Predictor` sa metodama `predict(texts) -> labels` i `predict_proba(texts) -> probabilities`. Modul za evaluaciju i demo aplikacija koriste isključivo ovaj interfejs, pa dodavanje novog modela ne zahteva izmene na drugim mestima.

## Tehnologije

Tehnološki stek se zasniva na Python ekosistemu, sa alatima koji su standard u ML timovima.

| Oblast | Tehnologija | Namena |
| --- | --- | --- |
| Jezik i okruženje | Python 3.11, uv ili pip sa `pyproject.toml` | Upravljanje zavisnostima i virtuelnim okruženjem |
| Obrada podataka | pandas, pyarrow | Učitavanje, čišćenje i čuvanje u Parquet formatu |
| Normalizacija teksta | `cyrtranslit` ili sopstvena mapa ćirilica u latinicu, `regex` | Ujednačavanje pisma i dijakritika |
| Vizualizacija | matplotlib, seaborn | Grafikoni za EDA i izveštaj |
| Klasični ML | scikit-learn, Optuna | TF-IDF, logistička regresija, SVM, Naive Bayes, pretraga hiperparametara |
| Neuronska mreža | PyTorch, fastText vektori za srpski (`cc.sr.300`) | BiLSTM i ručna trening petlja |
| Transformer | Hugging Face transformers, datasets, evaluate, `classla/bcms-bertic` | Fino podešavanje i inferencija |
| Interpretacija | koeficijenti modela, opciono LIME ili SHAP, attention vizualizacija | Objašnjenje predikcija |
| Praćenje eksperimenata | MLflow (lokalni tracking server) | Beleženje parametara, metrika i artefakata |
| Konfiguracija | YAML, pydantic | Validacija konfiguracionih fajlova |
| Testiranje i kvalitet | pytest, ruff, pre-commit | Unit testovi i formatiranje |
| Demo | Gradio, Hugging Face Spaces | Interaktivni prikaz i javni link |
| Trening na GPU-u | Google Colab ili Kaggle Notebooks | Besplatan GPU za transformer |
| Verzionisanje | Git, GitHub, GitHub Actions | Kontrola verzija i automatsko pokretanje testova |

## Plan implementacije i kriterijumi prihvatanja

Implementacija se odvija u sedam faza tokom dve do tri nedelje. Svaka faza se završava tek kada su ispunjeni njeni kriterijumi prihvatanja.

1. **Postavka projekta (dan 1).** Kreiranje strukture repozitorijuma, `pyproject.toml`, Makefile-a, pre-commit konfiguracije i GitHub Actions toka za testove.
   - Kriterijum prihvatanja. Komande `make install` i `make test` se izvršavaju bez greške na čistom okruženju.
2. **Podaci i čišćenje (dani 2 i 3).** Implementacija modula `data` i skripte `prepare_data.py`.
   - Kriterijum prihvatanja. Postoje tri Parquet fajla sa stratifikovanom podelom, izveštaj `data_quality.md` je generisan, a unit testovi pokrivaju normalizaciju ćirilice, dijakritika i uklanjanje duplikata.
3. **Eksplorativna analiza (dan 4).** Implementacija modula `eda` i notebook-a `01_eda.ipynb`.
   - Kriterijum prihvatanja. Izveštaj sadrži najmanje tri konkretna, numerički potkrepljena uvida o podacima.
4. **Klasični modeli (dani 5 i 6).** Implementacija Pipeline-a, pretrage hiperparametara i MLflow beleženja.
   - Kriterijum prihvatanja. Najbolji klasični model je sačuvan, macro F1 na validacionom skupu je zabeležen, a lista od 20 najuticajnijih reči po klasi je izvezena.
5. **Neuronska mreža (dani 7 do 9).** Implementacija tokenizer-a, Dataset klase, BiLSTM modela i trening petlje.
   - Kriterijum prihvatanja. Krive gubitka za trening i validaciju su sačuvane, early stopping radi, a model se učitava preko interfejsa `Predictor`.
6. **Transformer (dani 10 do 12).** Implementacija skripte za fino podešavanje i Colab notebook-a.
   - Kriterijum prihvatanja. Fino podešen model je objavljen na Hugging Face Hub-u i učitava se lokalno preko interfejsa `Predictor`.
7. **Evaluacija, demo i dokumentacija (dani 13 do 15).** Implementacija modula `evaluation`, Gradio aplikacije i finalnog README fajla.
   - Kriterijum prihvatanja. Zbirna tabela poređenja postoji, analiza grešaka sadrži najmanje 30 ručno pregledanih primera, demo je dostupan na javnom linku, a README sadrži rezultate i uputstvo za pokretanje.

Konačna tabela rezultata u README fajlu ima sledeći oblik, a vrednosti se popunjavaju posle evaluacije.

| Model | Macro F1 | Accuracy | Vreme treninga | Inferencija po primeru | Veličina modela |
| --- | --- | --- | --- | --- | --- |
| TF-IDF i logistička regresija |  |  |  |  |  |
| TF-IDF i linearni SVM |  |  |  |  |  |
| BiLSTM |  |  |  |  |  |
| BERTić |  |  |  |  |  |
