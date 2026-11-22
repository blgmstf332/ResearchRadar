
-- ResearchRadar V5 PostgreSQL veritabani semasi
-- 500 topic, 5000 topic-year, 119229 work, 5000 score rows

-- 1. Arastirma konulari
CREATE TABLE IF NOT EXISTS topics (
    topic_id TEXT PRIMARY KEY,
    topic_name TEXT NOT NULL,
    domain TEXT,
    field TEXT,
    subfield TEXT,
    works_count BIGINT CHECK (works_count >= 0)
);

-- 2. Yillik gercek yayin sayilari
CREATE TABLE IF NOT EXISTS topic_year_counts (
    topic_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    publication_count BIGINT NOT NULL
        CHECK (publication_count >= 0),
    retrieved_at_utc TIMESTAMPTZ,

    PRIMARY KEY (topic_id, year),

    FOREIGN KEY (topic_id)
        REFERENCES topics(topic_id)
);

-- 3. Temizlenmis makaleler
CREATE TABLE IF NOT EXISTS works (
    work_id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    publication_year INTEGER,
    publication_date DATE,
    title TEXT,
    doi TEXT,
    type TEXT,
    cited_by_count INTEGER,
    abstract TEXT,
    has_abstract BOOLEAN,
    counts_by_year_json JSONB,
    is_retracted BOOLEAN,
    missing_title BOOLEAN,
    missing_doi BOOLEAN,
    missing_abstract BOOLEAN,

    FOREIGN KEY (topic_id)
        REFERENCES topics(topic_id)
);

-- 4. Konu-yil bazli Emergence Score ve ozellikleri
CREATE TABLE IF NOT EXISTS topic_year_scores (
    topic_id TEXT NOT NULL,
    year INTEGER NOT NULL,

    prev_year_count BIGINT,
    count_2y_ago BIGINT,
    count_3y_ago BIGINT,

    growth_1y_pct DOUBLE PRECISION,
    growth_2y_pct DOUBLE PRECISION,
    growth_3y_pct DOUBLE PRECISION,
    growth_acceleration DOUBLE PRECISION,
    growth_stability DOUBLE PRECISION,

    citation_sample_size INTEGER,
    citation_velocity_1y DOUBLE PRECISION,
    citation_velocity_median DOUBLE PRECISION,
    early_citation_mean DOUBLE PRECISION,

    growth_score DOUBLE PRECISION,
    citation_score DOUBLE PRECISION,
    research_emergence_score DOUBLE PRECISION,
    is_emerging_candidate BOOLEAN,

    PRIMARY KEY (topic_id, year),

    FOREIGN KEY (topic_id, year)
        REFERENCES topic_year_counts(topic_id, year)
);

-- Sorgulari hizlandirmak icin indeksler
CREATE INDEX IF NOT EXISTS idx_works_topic_id
    ON works(topic_id);

CREATE INDEX IF NOT EXISTS idx_works_publication_year
    ON works(publication_year);

CREATE INDEX IF NOT EXISTS idx_scores_emergence
    ON topic_year_scores(research_emergence_score);
