# EDA Report: Student Loan Complaints

## Overview

This report summarizes the exploratory analysis conducted on a dataset of consumer complaints related to student loans, sourced from the Consumer Financial Protection Bureau (CFPB). The broader goal of this project is to automatically extract the core problem (Issue) and the specific sub-problem (Sub-issue) from a consumer's complaint narrative, eliminating the need for manual tagging. Before building any model, a thorough understanding of the data was necessary to make informed decisions about what to keep, what to discard, and what challenges to expect.

## The Dataset

The dataset contains approximately 52,988 records and 16 features, covering complaints submitted between 2023 and early 2026. Most columns are categorical and describe metadata about each complaint, such as the company involved, the submission channel, the state of the consumer, and how the company responded. The free-text field, Consumer complaint narrative, is the most valuable column for our purposes as it contains the actual description written by the consumer. However, it is missing in roughly half of all records, leaving 25,603 usable samples after filtering.

Several columns were identified as too sparse or uninformative to be useful. Tags and Company public response are missing in over 74% and 90% of records respectively. The Product column contains only one unique value across the entire dataset and was dropped entirely.

## Complaint Trends and Key Players

Complaint volume grew sharply between 2023 and 2025, more than doubling over that period, with a notable spike in early 2025 reaching nearly 3,500 complaints in a single month. This anomaly likely correlates with the resumption of federal student loan payments following a prolonged pause, which created widespread confusion and operational pressure on servicers.

The complaints are heavily concentrated among a small number of federal loan servicers, with MOHELA accounting for nearly 20,000 complaints alone, followed by Nelnet, Inc. It is worth noting that these are servicers rather than lenders. Their role is administrative: managing billing, customer service, and payment processing on behalf of the federal government. The high complaint volumes suggest that consumer frustration stems primarily from poor administration and communication rather than from the loans themselves.

A deeper look at compliance revealed two distinct crisis periods. EdFinancial Services experienced a near-complete compliance collapse between mid-2023 and mid-2024, with untimely response rates approaching 100%, before recovering abruptly. MOHELA then entered its own crisis from early 2025 onward, sustaining untimely rates of 70 to 85% with no sign of recovery through 2026. The fact that these two crises do not overlap points to systemic issues tied to federal servicing transitions rather than industry-wide failures.

## Understanding the Target Variables

The two labels we aim to predict automatically are Issue and Sub-issue. The dataset contains 12 unique Issues and 52 unique Sub-issues, forming a hierarchical structure where each Sub-issue belongs to a parent Issue. The distribution of these labels is heavily skewed: the most common Issue, "Dealing with your lender or servicer", accounts for over 30,000 complaints, more than three times the volume of the second category. This imbalance carries over into the sub-issue level as well and will need to be addressed during model training through techniques such as class weighting and stratified sampling.

## Why Metadata Cannot Replace the Narrative

One of the most important findings of this analysis is that the complaint metadata offers virtually no predictive power for determining the Issue or Sub-issue. A statistical correlation analysis using Cramér's V confirmed that columns such as Company, State, and submission channel all score below 0.25 in association with the target labels. The only strong association found (0.95) is between Issue and Sub-issue themselves, which is expected given their hierarchical relationship.

This finding is significant because it rules out any shortcut. The only way to reliably tag a complaint is to read what the consumer actually wrote. The narrative is not just the best feature available; it is the only one that matters.

## Text Length Filtering

Before exporting the final dataset, a length-based filtering step was applied to the complaint narratives. After removing records with missing text, complaints shorter than the 25th percentile character count were dropped. These very short texts were found to lack sufficient context for reliable classification and would introduce noise into the model. This step is a deliberate quality gate: it trades a small reduction in dataset size for a meaningful improvement in the signal-to-noise ratio of the training data.

## Conclusion

The analysis confirmed that the dataset is complex, imbalanced, and heavily dependent on free text for meaningful signal. The metadata explored throughout this report was ultimately set aside, but examining it was not wasted effort. It provided context about who is complaining, about whom, and when, and it gave statistical grounding to the decision to rely exclusively on the consumer narrative. The clean dataset exported at the end of this phase contains records with the narrative, Issue, and Sub-issue that have passed both null filtering and the 25th percentile length threshold, and is ready for the next stage of the pipeline.
