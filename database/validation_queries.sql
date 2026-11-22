
-- ResearchRadar V5
-- PostgreSQL veri dogrulama sorgulari
-- Bu sorgular veri degistirmez.

-- 1. Tablolardaki toplam kayit sayilari
SELECT 'topics' AS table_name, COUNT(*) AS total_rows
FROM topics

UNION ALL

SELECT 'topic_year_counts', COUNT(*)
FROM topic_year_counts

UNION ALL

SELECT 'works', COUNT(*)
FROM works

UNION ALL

SELECT 'topic_year_scores', COUNT(*)
FROM topic_year_scores;


-- 2. Tekrarli anahtar kontrolleri
-- Her iki sayinin da ayni olmasini bekliyoruz.
SELECT
    'topics' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT topic_id) AS unique_keys
FROM topics

UNION ALL

SELECT
    'topic_year_counts',
    COUNT(*),
    COUNT(DISTINCT (topic_id, year))
FROM topic_year_counts

UNION ALL

SELECT
    'works',
    COUNT(*),
    COUNT(DISTINCT work_id)
FROM works

UNION ALL

SELECT
    'topic_year_scores',
    COUNT(*),
    COUNT(DISTINCT (topic_id, year))
FROM topic_year_scores;


-- 3. Hesaplanmis ve NULL Emergence Score sayilari
SELECT
    COUNT(*) AS total_score_rows,
    COUNT(research_emergence_score) AS calculated_scores,
    COUNT(*) FILTER (
        WHERE research_emergence_score IS NULL
    ) AS null_scores
FROM topic_year_scores;


-- 4. Konu-yil kayitlarinin topics tablosuyla iliskisi
-- Beklenen: 0
SELECT COUNT(*) AS missing_topic_references
FROM topic_year_counts AS yc
LEFT JOIN topics AS t
    ON yc.topic_id = t.topic_id
WHERE t.topic_id IS NULL;


-- 5. Skor kayitlarinin konu-yil tablosuyla iliskisi
-- Beklenen: 0
SELECT COUNT(*) AS missing_year_references
FROM topic_year_scores AS s
LEFT JOIN topic_year_counts AS yc
    ON s.topic_id = yc.topic_id
    AND s.year = yc.year
WHERE yc.topic_id IS NULL;


-- 6. Makalelerin topics tablosuyla iliskisi
-- Works aktarimi tamamlanana kadar da 0 olacaktir.
SELECT COUNT(*) AS missing_work_topic_references
FROM works AS w
LEFT JOIN topics AS t
    ON w.topic_id = t.topic_id
WHERE t.topic_id IS NULL;


-- 7. Yillara gore yayin ve skor kapsami
SELECT
    yc.year,
    COUNT(*) AS topic_year_rows,
    COUNT(s.topic_id) AS score_rows,
    COUNT(s.research_emergence_score) AS calculated_scores,
    COUNT(*) FILTER (
        WHERE s.research_emergence_score IS NULL
    ) AS missing_scores
FROM topic_year_counts AS yc
LEFT JOIN topic_year_scores AS s
    ON yc.topic_id = s.topic_id
    AND yc.year = s.year
GROUP BY yc.year
ORDER BY yc.year;
