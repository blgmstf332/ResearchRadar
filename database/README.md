
# ResearchRadar PostgreSQL Database

ResearchRadar V5 veri setinin PostgreSQL'e aktarimi.

## Gereksinimler

- PostgreSQL 17
- Python 3
- psycopg 3

Python bagimliligini yuklemek icin:

`python -m pip install "psycopg[binary]"`

## Veritabani olusturma

PostgreSQL uzerinde `researchradar` adli bir veritabani olusturun.

PowerShell ornegi:

`& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -p 5432 -d postgres -c "CREATE DATABASE researchradar;"`

## Tablolari olusturma

Proje ana klasorunden:

`& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -p 5432 -d researchradar -v ON_ERROR_STOP=1 -f "database/schema.sql"`

Bu komut dort tabloyu olusturur:

- topics
- topic_year_counts
- works
- topic_year_scores

## CSV dosyalari

Kucuk CSV dosyalari `data/samples/` altindadir:

- topic_candidates_500.csv
- topic_year_counts_500.csv
- topic_year_scores_v2_500.csv

Buyuk makale dosyasi Google Drive'dan alinmalidir:

- work_details_500_topics_n30_cleaned.csv

Bu dosya `data/processed/` altina yerlestirilmelidir.

Buyuk CSV dosyasini GitHub'a eklemeyin.

## Veri aktarimi

Once konu, konu-yil ve skor verilerini aktar:

`python database/import_data.py`

Ardindan makaleleri aktar:

`python database/import_works.py`

Scriptler PostgreSQL sifresini calisma sirasinda ister.
Sifre kodun icine yazilmaz.

Aktarimlar UPSERT kullandigi icin ayni veri tekrar
calistirildiginda kayitlar cogalmaz.

Yeni veri setleri icin import_data.py dosyasinin
--topics, --counts ve --scores parametreleri
kullanilabilir.

Makaleler icin import_works.py dosyasindaki
CSV_PATH ayari yeni dosya yoluna gore duzenlenmelidir.

Not: UPSERT mevcut kayitlari gunceller ancak yeni
CSV'den kaldirilmis eski kayitlari otomatik silmez.

## Veri dogrulama

`& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -p 5432 -d researchradar -v ON_ERROR_STOP=1 -f "database/validation_queries.sql"`

Bu sorgular kayit sayilarini, benzersiz anahtarlari,
NULL skor degerlerini ve tablo iliskilerini kontrol eder.

## V5 dogrulanmis sonuc

- 500 konu
- 5.000 konu-yil kaydi
- 119.229 benzersiz makale
- 5.000 skor tablosu satiri
- 3.488 hesaplanmis Emergence Score
- 1.512 NULL Emergence Score

2016-2018 yillarinda hesaplanmis Emergence Score yoktur.

Eksik skorlar 0 yerine NULL olarak saklanir.

Gercek yillik yayin sayilari topic_year_counts
tablosundan alinmalidir; orneklenmis works
tablosundan hesaplanmamalidir.

## Guvenlik

PostgreSQL sifrelerini, baglanti sirlarini ve buyuk
CSV dosyalarini GitHub'a gondermeyin.
