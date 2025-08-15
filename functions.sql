CREATE OR REPLACE FUNCTION skm.demo.get_top_side_effects(drug STRING)
  RETURNS TABLE(side_effect STRING, report_count BIGINT)
  RETURN
    SELECT
      r.pt AS side_effect, -- preferred term of the side effect
      COUNT(*) AS report_count
    FROM
      skm.demo.drug d
        JOIN skm.demo.reac r
          ON d.primaryid = r.primaryid
    WHERE
      d.role_cod = 'PS' -- Primary Suspect Drug
      AND LOWER(d.drugname) = LOWER(drug)
    GROUP BY
      r.pt
    ORDER BY
      report_count DESC
    LIMIT 10;

-- when 2 drugs are given to same patient, return top 10 side effects    
CREATE OR REPLACE FUNCTION skm.demo.get_drug_pair_interactions(drug1 STRING, drug2 STRING)
  RETURNS TABLE(shared_side_effect STRING, shared_report_count BIGINT)
  RETURN
    SELECT
      r.pt AS shared_side_effect,
      COUNT(DISTINCT d1.primaryid) AS shared_report_count
    FROM
      skm.demo.drug d1
        JOIN skm.demo.drug d2
          ON d1.primaryid = d2.primaryid
        JOIN skm.demo.reac r
          ON d1.primaryid = r.primaryid
    WHERE
      LOWER(d1.drugname) = LOWER(drug1)
      AND LOWER(d2.drugname) = LOWER(drug2)
    GROUP BY
      r.pt
    ORDER BY
      shared_report_count DESC
    LIMIT 10;
      
-- for similar age group and sex code, get top side effects      
CREATE OR REPLACE FUNCTION skm.demo.get_side_effect_by_demographics(
    drug STRING, age_group STRING, sex_code STRING
  )
  RETURNS TABLE(side_effect STRING, report_count BIGINT)
  RETURN
    SELECT
      r.pt AS side_effect,
      COUNT(*) AS report_count
    FROM
      skm.demo.drug d
        JOIN skm.demo.reac r
          ON d.primaryid = r.primaryid
        JOIN skm.demo.demographics dem
          ON d.primaryid = dem.primaryid
    WHERE
      d.role_cod = 'PS'
      AND LOWER(d.drugname) = LOWER(drug)
      AND dem.age_grp = age_group
      AND dem.sex = sex_code
    GROUP BY
      r.pt
    ORDER BY
      report_count DESC
    LIMIT 10 ;

-- Find severe side effects for a drug
CREATE OR REPLACE FUNCTION skm.demo.get_severe_side_effects(drug STRING)
  RETURNS TABLE(side_effect STRING, outcome_code STRING, report_count BIGINT)
  RETURN
    SELECT
      r.pt AS side_effect,
      o.outc_cod AS outcome_code,
      COUNT(*) AS report_count
    FROM
      skm.demo.drug d
        JOIN skm.demo.reac r
          ON d.primaryid = r.primaryid
        JOIN skm.demo.outcome o
          ON d.primaryid = o.primaryid
    WHERE
      d.role_cod = 'PS' -- Primary Suspect Drug
      AND LOWER(d.drugname) = LOWER(drug)
      AND o.outc_cod IN ('DE', 'LT', 'HO', 'DS') -- Death, Life Threatening, Hospitalized, Disability.
    GROUP BY
      r.pt,
      o.outc_cod
    ORDER BY
      report_count DESC
    LIMIT 10;            