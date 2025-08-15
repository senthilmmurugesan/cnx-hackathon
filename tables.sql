CREATE TABLE skm.demo.demographics (
  primaryid STRING COMMENT 'A unique identifier for each case report, allowing for easy tracking and reference of individual cases.',
  caseid STRING COMMENT 'An identifier specific to the case, which may be used to group related events and reports together.',
  caseversion STRING COMMENT 'Indicates the version of the case report, useful for tracking updates or changes made to the case over time.',
  i_f_code STRING COMMENT 'Represents the international classification code for the case, providing a standardized way to categorize the type of case.',
  event_dt STRING COMMENT 'The date when the event related to the case occurred, essential for understanding the timeline of events.',
  mfr_dt STRING COMMENT 'The date when the manufacturer reported the case, which can help in analyzing reporting timelines.',
  init_fda_dt STRING COMMENT 'The initial date when the case was reported to the FDA, important for tracking the start of regulatory oversight.',
  fda_dt STRING COMMENT 'The date when the FDA received the report, providing insight into the regulatory process timeline.',
  rept_cod STRING COMMENT 'A code that indicates the type of report, which can be useful for categorizing the nature of the case.',
  auth_num STRING COMMENT 'The authorization number associated with the case, which may be relevant for regulatory or compliance purposes.',
  mfr_num STRING COMMENT 'The manufacturer number linked to the case, helping to identify the company responsible for the product involved.',
  mfr_sndr STRING COMMENT 'The sender of the manufacturer report, providing information about who submitted the case to the regulatory body.',
  lit_ref STRING COMMENT 'References to literature or studies related to the case, which can provide additional context or background information.',
  age STRING COMMENT 'The age of the individual involved in the case, which is crucial for demographic analysis.',
  age_cod STRING COMMENT 'A code representing the age category of the individual, facilitating easier demographic segmentation.',
  age_grp STRING COMMENT 'The age group classification for the individual, useful for analyzing trends across different age demographics.',
  sex STRING COMMENT 'The sex of the individual involved in the case, important for understanding demographic patterns.',
  e_sub STRING COMMENT 'Indicates whether the case is a submission for an event, which can help in categorizing the nature of the report.',
  wt STRING COMMENT 'The weight of the individual involved in the case, which may be relevant for certain types of analyses.',
  wt_cod STRING COMMENT 'A code representing the weight category of the individual, aiding in demographic analysis.',
  rept_dt STRING COMMENT 'The date when the case was reported, essential for tracking the reporting timeline.',
  to_mfr STRING COMMENT 'Indicates whether the report was sent to the manufacturer, which can be important for understanding the flow of information.',
  occp_cod STRING COMMENT 'A code representing the occupation of the individual, useful for analyzing trends related to occupational exposure.',
  reporter_country STRING COMMENT 'The country where the report originated, providing context for geographical trends in case reporting.',
  occr_country STRING COMMENT 'The country where the event occurred, which can be important for understanding the geographical distribution of cases.')
USING delta
COMMENT 'The table contains data related to case reports, including details about the case, events, and demographics of individuals involved. It can be used for analyzing trends in case reporting, understanding the demographics of reported cases, and tracking the timeline of events related to each case. Key information includes case identifiers, event dates, manufacturer details, and demographic data such as age and sex.'
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.feature.appendOnly' = 'supported',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.invariants' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7')

CREATE TABLE skm.demo.drug (
  primaryid STRING COMMENT 'A unique identifier for each record in the dataset, allowing for easy reference and tracking of individual cases.',
  caseid STRING COMMENT 'Identifies the specific case associated with the record, which can be useful for linking related information.',
  drug_seq STRING COMMENT 'Represents the sequence number of the drug in the context of the case, indicating the order of administration or reporting.',
  role_cod STRING COMMENT 'Indicates the role of the individual involved in the case, such as prescriber, patient, or healthcare provider.',
  drugname STRING COMMENT 'The name of the drug being reported in the case, essential for understanding the context of the data.',
  prod_ai STRING COMMENT 'Contains information about the product\'s active ingredient, which is crucial for identifying the therapeutic component of the drug.',
  val_vbm STRING COMMENT 'Represents the value of the variable being measured, which can provide insights into the drug\'s effects or outcomes.',
  route STRING COMMENT 'Describes the method of administration for the drug, such as oral, intravenous, or topical, which is important for understanding its use.',
  dose_vbm STRING COMMENT 'Indicates the value of the dose administered, providing critical information for evaluating treatment regimens.',
  cum_dose_chr STRING COMMENT 'Represents the cumulative dose characteristic, which can help assess the total exposure to the drug over time.',
  cum_dose_unit STRING COMMENT 'Specifies the unit of measurement for the cumulative dose, ensuring clarity in dosage reporting.',
  dechal STRING COMMENT 'Indicates whether the drug was discontinued, which is important for understanding treatment outcomes and adverse events.',
  rechal STRING COMMENT 'Indicates whether the drug was reintroduced after being discontinued, providing insights into treatment decisions.',
  lot_num STRING COMMENT 'The lot number of the drug, which is essential for tracking and tracing products in case of recalls or safety concerns.',
  exp_dt STRING COMMENT 'The expiration date of the drug, which is important for ensuring the safety and efficacy of the medication.',
  nda_num STRING COMMENT 'The New Drug Application number, which is crucial for regulatory tracking and understanding the drug\'s approval status.',
  dose_amt STRING COMMENT 'Specifies the amount of the drug administered, providing essential information for evaluating treatment effectiveness.',
  dose_unit STRING COMMENT 'Indicates the unit of measurement for the dose amount, ensuring accurate interpretation of dosing information.',
  dose_form STRING COMMENT 'Describes the physical form of the drug, such as tablet, liquid, or injection, which can influence administration and patient compliance.',
  dose_freq STRING COMMENT 'Indicates how often the drug is administered, which is critical for understanding treatment schedules and adherence.')
USING delta
COMMENT 'The table contains data related to drug administration cases. It includes information such as the drug name, dosage details, administration route, and case identifiers. This data can be used for analyzing drug usage patterns, monitoring compliance with dosing guidelines, and evaluating the effectiveness of treatments.'
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.feature.appendOnly' = 'supported',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.invariants' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7')

CREATE TABLE skm.demo.outcome (
  primaryid STRING COMMENT 'Represents a unique identifier for each record in the dataset, ensuring that every entry can be distinctly referenced.',
  caseid STRING COMMENT 'Serves as a unique identifier for specific cases, allowing for easy tracking and management of individual cases within the dataset.',
  outc_cod STRING COMMENT 'Contains codes that indicate the outcomes associated with each case, providing insight into the results and effectiveness of the cases.')
USING delta
COMMENT 'The table contains data related to case outcomes. It includes identifiers for each case and their corresponding outcome codes. This data can be used to track case progress, analyze outcome trends, and support reporting on case management effectiveness.'
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.feature.appendOnly' = 'supported',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.invariants' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7')

CREATE TABLE skm.demo.reac (
  primaryid STRING COMMENT 'Represents a unique identifier for each record in the dataset, ensuring that every entry can be distinctly referenced.',
  caseid STRING COMMENT 'Identifies the specific case associated with the record, allowing for tracking and management of individual cases.',
  pt STRING COMMENT 'Contains the preferred term related to the case, which provides a standardized description of the condition or event being recorded.',
  drug_rec_act STRING COMMENT 'Details the actions taken regarding drug recommendations, which can include prescribing, administering, or monitoring drug therapies.')
USING delta
COMMENT 'The table contains data related to drug recommendations and actions associated with specific cases. It includes identifiers for each case and the corresponding drug recommendation actions. This data can be used for tracking drug-related cases, analyzing the effectiveness of drug recommendations, and understanding patterns in drug usage across different cases.'
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.feature.appendOnly' = 'supported',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.invariants' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7')
